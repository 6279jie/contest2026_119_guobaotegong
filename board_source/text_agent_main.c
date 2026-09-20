/****************************************************************************
 * packages/text_agent/text_agent_main.c
 *
 * Minimal HTTP text receiver. It intentionally has no model runtime,
 * tool registry, session store, WebSocket server, audio path, or heap-backed
 * message queue. One request is handled at a time with fixed-size buffers.
 ****************************************************************************/

#include <nuttx/config.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef CONFIG_EXAMPLES_TEXT_AGENT_PORT
#  define CONFIG_EXAMPLES_TEXT_AGENT_PORT 28789
#endif

#define TEXT_AGENT_HEADER_MAX 512
#define TEXT_AGENT_TEXT_MAX   256
#define TEXT_AGENT_REQUEST_MAX (TEXT_AGENT_HEADER_MAX + TEXT_AGENT_TEXT_MAX + 1)

#define TEXT_AGENT_QUEUE_NAME "/tmp/camera_lcd_text"

static mqd_t g_text_agent_queue = (mqd_t)-1;

static void text_agent_publish(const char *text)
{
  char discarded[64];
  char message[64];
  size_t length;

  if (g_text_agent_queue == (mqd_t)-1)
    {
      return;
    }

  length = strlen(text);
  if (length >= sizeof(message))
    {
      length = sizeof(message) - 1;
    }

  memcpy(message, text, length);
  message[length] = '\0';

  if (mq_send(g_text_agent_queue, message, length + 1, 0) == 0)
    {
      return;
    }

  /* Latest text wins. The camera never receives an unbounded backlog. */
  if (errno == EAGAIN &&
      mq_receive(g_text_agent_queue, discarded, sizeof(discarded), NULL) >= 0)
    {
      if (mq_send(g_text_agent_queue, message, length + 1, 0) < 0)
        {
          printf("text_agent: publish failed: %d\\n", errno);
        }
    }
  else if (errno != EAGAIN)
    {
      printf("text_agent: publish failed: %d\\n", errno);
    }
}

static void text_agent_reply(int fd, int status, const char *reason,
                             const char *body)
{
  char response[256];
  int length;

  length = snprintf(response, sizeof(response),
                    "HTTP/1.1 %d %s\r\n"
                    "Content-Type: text/plain; charset=utf-8\r\n"
                    "Content-Length: %u\r\n"
                    "Connection: close\r\n\r\n%s",
                    status, reason, (unsigned int)strlen(body), body);
  if (length > 0)
    {
      send(fd, response, (size_t)length, 0);
    }
}

static int text_agent_content_length(const char *headers)
{
  const char *line = headers;

  while (line != NULL && *line != '\0')
    {
      if (strncasecmp(line, "Content-Length:", 15) == 0)
        {
          char *endptr;
          long value;

          line += 15;
          while (*line == ' ' || *line == '\t')
            {
              line++;
            }

          value = strtol(line, &endptr, 10);
          if (endptr == line || value <= 0 ||
              value > TEXT_AGENT_TEXT_MAX)
            {
              return -1;
            }

          return (int)value;
        }

      line = strstr(line, "\r\n");
      if (line != NULL)
        {
          line += 2;
        }
    }

  return -1;
}

static int text_agent_extract(const char *body, size_t body_len,
                              char *text, size_t text_size)
{
  const char *start = body;
  const char *end = body + body_len;
  const char *key;
  size_t length;

  while (start < end && (*start == ' ' || *start == '\t' ||
                         *start == '\r' || *start == '\n'))
    {
      start++;
    }

  if (start < end && *start == '{')
    {
      key = strstr(start, "\"text\"");
      if (key == NULL)
        {
          return -1;
        }

      key = strchr(key + 6, ':');
      if (key == NULL)
        {
          return -1;
        }

      key++;
      while (key < end && (*key == ' ' || *key == '\t'))
        {
          key++;
        }

      if (key >= end || *key != '\"')
        {
          return -1;
        }

      start = ++key;
      key = strchr(start, '\"');
      if (key == NULL || key > end)
        {
          return -1;
        }

      end = key;
    }

  while (end > start && (end[-1] == ' ' || end[-1] == '\t' ||
                         end[-1] == '\r' || end[-1] == '\n'))
    {
      end--;
    }

  length = (size_t)(end - start);
  if (length == 0 || length >= text_size)
    {
      return -1;
    }

  memcpy(text, start, length);
  text[length] = '\0';
  return (int)length;
}

static void text_agent_handle(int fd)
{
  struct timeval timeout =
    {
      .tv_sec = 5,
      .tv_usec = 0
    };
  char request[TEXT_AGENT_REQUEST_MAX];
  char text[TEXT_AGENT_TEXT_MAX];
  char *header_end;
  size_t received = 0;
  size_t header_size;
  size_t body_received;
  int content_length;
  ssize_t count;

  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  memset(request, 0, sizeof(request));

  while (received < TEXT_AGENT_HEADER_MAX)
    {
      count = recv(fd, request + received,
                   sizeof(request) - 1 - received, 0);
      if (count <= 0)
        {
          text_agent_reply(fd, 408, "Request Timeout", "timeout\n");
          return;
        }

      received += (size_t)count;
      request[received] = '\0';
      header_end = strstr(request, "\r\n\r\n");
      if (header_end != NULL)
        {
          break;
        }
    }

  header_end = strstr(request, "\r\n\r\n");
  if (header_end == NULL)
    {
      text_agent_reply(fd, 404, "Not Found", "use POST /text\n");
      return;
    }

  if (strncmp(request, "GET /text/", 10) == 0)
    {
      char *start = request + 10;
      char *end = strchr(start, ' ');
      if (end == NULL ||
          text_agent_extract(start, (size_t)(end - start),
                             text, sizeof(text)) < 0)
        {
          text_agent_reply(fd, 400, "Bad Request", "invalid text\n");
          return;
        }

      printf("TEXT_AGENT: %s\n", text);
      text_agent_publish(text);
      text_agent_reply(fd, 200, "OK", "received\n");
      return;
    }

  if (strncmp(request, "POST /text ", 11) != 0)
    {
      text_agent_reply(fd, 404, "Not Found", "use POST /text\n");
      return;
    }

  content_length = text_agent_content_length(request);
  if (content_length <= 0 || content_length > TEXT_AGENT_TEXT_MAX)
    {
      text_agent_reply(fd, 400, "Bad Request", "invalid Content-Length\n");
      return;
    }

  header_size = (size_t)(header_end + 4 - request);
  body_received = received > header_size ? received - header_size : 0;
  if (body_received > (size_t)content_length)
    {
      body_received = (size_t)content_length;
    }

  while (body_received < (size_t)content_length)
    {
      count = recv(fd, request + header_size + body_received,
                   (size_t)content_length - body_received, 0);
      if (count <= 0)
        {
          text_agent_reply(fd, 400, "Bad Request", "incomplete body\n");
          return;
        }

      body_received += (size_t)count;
    }

  if (text_agent_extract(request + header_size, body_received,
                         text, sizeof(text)) < 0)
    {
      text_agent_reply(fd, 400, "Bad Request", "invalid text\n");
      return;
    }

  printf("TEXT_AGENT: %s\n", text);
  text_agent_publish(text);
  text_agent_reply(fd, 200, "OK", "received\n");
}

int main(int argc, FAR char *argv[])
{
  struct sockaddr_in address;
  int server_fd;
  char packet[TEXT_AGENT_TEXT_MAX + 1];
  char text[TEXT_AGENT_TEXT_MAX];
  ssize_t length;

  (void)argc;
  struct mq_attr attributes;

  (void)argv;

  memset(&attributes, 0, sizeof(attributes));
  attributes.mq_maxmsg = 1;
  attributes.mq_msgsize = 64;
  g_text_agent_queue = mq_open(TEXT_AGENT_QUEUE_NAME,
                               O_RDWR | O_CREAT | O_NONBLOCK,
                               0644, &attributes);
  if (g_text_agent_queue == (mqd_t)-1)
    {
      printf("text_agent: text queue open failed: %d\n", errno);
    }

  /* Keep text delivery off TCP.  The camera owns a long-lived TCP stream;
   * using another TCP listener on this NuttX/WiFi combination can stop that
   * stream.  Text is short and latest-wins, so UDP is the appropriate
   * bounded transport here. */
  server_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (server_fd < 0)
    {
      printf("text_agent: socket failed: %d\n", errno);
      return EXIT_FAILURE;
    }

  memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(CONFIG_EXAMPLES_TEXT_AGENT_PORT);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
      printf("text_agent: bind failed: %d\n", errno);
      close(server_fd);
      return EXIT_FAILURE;
    }

  printf("text_agent: listening for UDP text on port %d\n",
         CONFIG_EXAMPLES_TEXT_AGENT_PORT);

  for (;;)
    {
      length = recvfrom(server_fd, packet, sizeof(packet) - 1, 0, NULL, NULL);
      if (length < 0)
        {
          if (errno == EINTR)
            {
              continue;
            }

          printf("text_agent: recvfrom failed: %d\n", errno);
          continue;
        }

      packet[length] = '\0';
      if (text_agent_extract(packet, (size_t)length, text, sizeof(text)) < 0)
        {
          printf("text_agent: invalid UDP text\n");
          continue;
        }

      printf("TEXT_AGENT: %s\n", text);
      text_agent_publish(text);
    }

  return EXIT_SUCCESS;
}
