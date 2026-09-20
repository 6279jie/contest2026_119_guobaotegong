/****************************************************************************
 * apps/examples/camera/camera_main.c
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.  The
 * ASF licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 ****************************************************************************/

/****************************************************************************
 * Included Files
 ****************************************************************************/

#include <nuttx/config.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

#include <nuttx/cache.h>
#include <nuttx/video/video.h>
#include <nuttx/video/v4l2_cap.h>

#include "camera_fileutil.h"
#include "camera_bkgd.h"
#include "camera_rgb565.h"
#include "camera_audio.h"
#include "camera_lcd_label.h"

#ifdef CONFIG_NETUTILS_WEBCLIENT
#  include "netutils/webclient.h"
#endif

/****************************************************************************
 * Pre-processor Definitions
 ****************************************************************************/

#define IMAGE_JPG_SIZE     (512*1024)  /* 512kB for FullHD Jpeg file. */
#define IMAGE_RGB_SIZE     (320*240*2) /* QVGA RGB565 */

#define HAND_ROI_SIZE      192
#define MODEL_INPUT_SIZE    96
#define LCD_PREVIEW_SIZE   240

#define VIDEO_BUFNUM       (3)
#define STILL_BUFNUM       (1)

/* The stream path owns a small fixed JPEG pool.  V4L2 buffers are copied into
 * this pool and immediately QBUF'ed, so a slow network cannot hold the camera
 * driver hostage. */
#define CAMERA_STREAM_QUEUE_DEPTH      (3)
#define CAMERA_STREAM_JPEG_SIZE        (256 * 1024)
#define CAMERA_STREAM_STACKSIZE        (16384)
#define CAMERA_STREAM_SEND_TIMEOUT_SEC (5)
#define CAMERA_STREAM_INTERVAL_MS      (0)
#define CAMERA_TEXT_UDP_PORT           (28789)
#define CAMERA_HELP_EVENT_UDP_PORT     (28790)
#define CAMERA_HELP_CONFIRM_TIMEOUT_SEC (5)
#define CAMERA_TEXT_MAX                (64)

#define MAX_CAPTURE_NUM     (100)
#define DEFAULT_CAPTURE_NUM (10)

#define START_CAPTURE_TIME  (5)   /* seconds */
#define KEEP_VIDEO_TIME     (10)  /* seconds */

#define APP_STATE_BEFORE_CAPTURE  (0)
#define APP_STATE_UNDER_CAPTURE   (1)
#define APP_STATE_AFTER_CAPTURE   (2)

#define CAMERA_DEV_PATH "/dev/video"

#ifdef CONFIG_NETUTILS_WEBCLIENT
#  define FRAME_UPLOAD_URL "http://192.168.1.182:8000/api/v1/frame"
#  define FRAME_HEALTH_URL "http://192.168.1.182:8000/health"
#endif

/****************************************************************************
 * Private Types
 ****************************************************************************/

struct v_buffer
{
  FAR uint32_t *start;
  uint32_t length;
  bool mapped;
};

typedef struct v_buffer v_buffer_t;

struct camera_stream_ctx_s
{
  pthread_t network_thread;
  bool network_started;
  bool network_joined;
  volatile bool capture_done;
  pthread_mutex_t lock;
  sem_t ready_sem;
  uint8_t *pool[CAMERA_STREAM_QUEUE_DEPTH];
  size_t pool_len[CAMERA_STREAM_QUEUE_DEPTH];
  uint8_t queue[CAMERA_STREAM_QUEUE_DEPTH];
  uint8_t queue_head;
  uint8_t queue_tail;
  uint8_t queue_count;
  bool pool_used[CAMERA_STREAM_QUEUE_DEPTH];
  unsigned int captured;
  unsigned int queued;
  unsigned int sent;
  unsigned int dropped;
  unsigned int bad_jpeg;
  unsigned int send_errors;
};

/****************************************************************************
 * Private Function Prototypes
 ****************************************************************************/

static int camera_prepare(int fd, enum v4l2_buf_type type,
                          uint32_t buf_mode, uint32_t pixformat,
                          uint16_t hsize, uint16_t vsize,
                          FAR struct v_buffer **vbuf,
                          uint8_t buffernum, int buffersize);
static int camera_prepare_mmap(int fd, enum v4l2_buf_type type,
                               uint32_t buf_mode, uint32_t pixformat,
                               uint16_t hsize, uint16_t vsize,
                               FAR struct v_buffer **vbuf,
                               uint8_t buffernum);
static void free_buffer(FAR struct v_buffer *buffers, uint8_t bufnum);
static void release_mmap_buffers(int fd, enum v4l2_buf_type type,
                                 FAR struct v_buffer *buffers,
                                 uint8_t bufnum);
static void reset_mmap_request(int fd, enum v4l2_buf_type type);
static int parse_arguments(int argc, FAR char *argv[],
                           FAR int *capture_num,
                           FAR enum v4l2_buf_type *type,
                           FAR bool *upload,
                           FAR bool *stream,
                           FAR bool *stream_forever,
                           FAR bool *network_benchmark,
                           FAR bool *network_stream_benchmark);
static int get_camimage(int fd, FAR struct v4l2_buffer *v4l2_buf,
                        enum v4l2_buf_type buf_type);
static int get_camimage_mmap(int fd, FAR struct v4l2_buffer *v4l2_buf,
                             enum v4l2_buf_type buf_type);
static int release_camimage(int fd, FAR struct v4l2_buffer *v4l2_buf);
static int start_stillcapture(int v_fd, enum v4l2_buf_type capture_type);
static int stop_stillcapture(int v_fd, enum v4l2_buf_type capture_type);
#ifdef CONFIG_NETUTILS_WEBCLIENT
static int camera_stream_init(FAR struct camera_stream_ctx_s *ctx);
static int camera_stream_start(FAR struct camera_stream_ctx_s *ctx);
static void camera_stream_stop(FAR struct camera_stream_ctx_s *ctx);
static void camera_stream_join(FAR struct camera_stream_ctx_s *ctx);
static int camera_stream_capture_and_queue(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type,
  FAR struct camera_stream_ctx_s *ctx,
  FAR struct v_buffer *buffers,
  bool mmap_mode);
static int upload_jpeg(FAR const uint8_t *data, size_t len);
static int network_stream_send_all(int sockfd, FAR const void *data,
                                   size_t len);
static int network_stream_open_connection(void);
static void network_stream_finish_connection(int sockfd);
static int network_stream_send_frame(int sockfd,
                                     FAR const uint8_t *data, size_t len,
                                     uint64_t frame_id,
                                     uint64_t timestamp_ms);
static int camera_text_udp_open(void);
static void camera_text_udp_poll(int sockfd);
static void camera_text_udp_close(int *sockfd);
static void camera_agent_handle_text(int sockfd, FAR const char *text);
static void camera_agent_send_event(int sockfd, FAR const char *event);
static int camera_network_benchmark(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type, int frame_count);
static int camera_network_stream_benchmark(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type, int frame_count);
#endif

/****************************************************************************
 * Private Data
 ****************************************************************************/
static uint16_t g_model_rgb565[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE]
  __attribute__((aligned(32)));
static uint8_t g_model_rgb888[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3]
  __attribute__((aligned(32)));
static bool g_model_rgb888_reported;

#ifdef CONFIG_NETUTILS_WEBCLIENT
static uint64_t g_upload_frame_id;
static bool g_camera_help_pending;
static time_t g_camera_help_deadline;
static struct sockaddr_in g_camera_text_sender;
static bool g_camera_text_sender_valid;

struct health_response_s
{
  char data[256];
  size_t len;
};

static void health_response_callback(FAR char **buffer, int offset,
                                     int datend, FAR int *buflen,
                                     FAR void *arg)
{
  FAR struct health_response_s *response = arg;
  size_t copylen;

  (void)buflen;

  if (datend <= offset || response->len >= sizeof(response->data) - 1)
    {
      return;
    }

  copylen = datend - offset;
  if (copylen > sizeof(response->data) - 1 - response->len)
    {
      copylen = sizeof(response->data) - 1 - response->len;
    }

  memcpy(response->data + response->len, *buffer + offset, copylen);
  response->len += copylen;
  response->data[response->len] = '\0';
}

static int sync_upload_frame_id(void)
{
  struct webclient_context ctx;
  struct health_response_s response;
  FAR char *value;
  FAR char *endptr;
  int ret;

  memset(&response, 0, sizeof(response));
  memset(&ctx, 0, sizeof(ctx));
  webclient_set_defaults(&ctx);
  ctx.method = "GET";
  ctx.url = FRAME_HEALTH_URL;
  ctx.buffer = response.data;
  ctx.buflen = sizeof(response.data) - 1;
  ctx.callback = health_response_callback;
  ctx.sink_callback_arg = &response;
  ctx.timeout_sec = 5;

  ret = webclient_perform(&ctx);
  if (ret < 0 || ctx.http_status < 200 || ctx.http_status >= 300)
    {
      printf("Frame ID sync failed: status=%u ret=%d\n",
             ctx.http_status, ret);
      return ERROR;
    }

  value = strstr(response.data, "\"latest_frame_id\"");
  if (value == NULL)
    {
      printf("Frame ID sync failed: latest_frame_id missing\n");
      return ERROR;
    }

  value = strchr(value, ':');
  if (value == NULL)
    {
      return ERROR;
    }

  value++;
  while (*value == ' ' || *value == '\t')
    {
      value++;
    }

  if (strncmp(value, "null", 4) == 0)
    {
      g_upload_frame_id = 0;
      return OK;
    }

  g_upload_frame_id = strtoull(value, &endptr, 10);
  if (endptr == value)
    {
      printf("Frame ID sync failed: invalid latest_frame_id\n");
      return ERROR;
    }

  printf("Frame ID sync: latest=%" PRIu64 "\n", g_upload_frame_id);
  return OK;
}

/****************************************************************************
 * Name: camera_prepare_mmap
 *
 * Description:
 *   Prepare a V4L2 capture stream using the driver's MMAP buffer heap.
 *   This follows the OpenVela camera tool pattern: request buffers, query
 *   their offsets, map them once, queue them, and then start streaming.
 ****************************************************************************/

static int camera_prepare_mmap(int fd, enum v4l2_buf_type type,
                               uint32_t buf_mode, uint32_t pixformat,
                               uint16_t hsize, uint16_t vsize,
                               FAR struct v_buffer **vbuf,
                               uint8_t buffernum)
{
  struct v4l2_format fmt;
  struct v4l2_requestbuffers req;
  struct v4l2_buffer buf;
  uint32_t cnt;
  int ret;

  memset(&fmt, 0, sizeof(fmt));
  fmt.type                = type;
  fmt.fmt.pix.width       = hsize;
  fmt.fmt.pix.height      = vsize;
  fmt.fmt.pix.field       = V4L2_FIELD_ANY;
  fmt.fmt.pix.pixelformat = pixformat;

  ret = ioctl(fd, VIDIOC_S_FMT, (uintptr_t)&fmt);
  if (ret < 0)
    {
      printf("Failed to VIDIOC_S_FMT(MMAP): errno = %d\n", errno);
      return ret;
    }

  memset(&req, 0, sizeof(req));
  req.type   = type;
  req.memory = V4L2_MEMORY_MMAP;
  req.count  = buffernum;
  req.mode   = buf_mode;

  ret = ioctl(fd, VIDIOC_REQBUFS, (uintptr_t)&req);
  if (ret < 0 || req.count == 0 || req.count > buffernum)
    {
      printf("Failed to VIDIOC_REQBUFS(MMAP): errno = %d count=%u\n",
             errno, (unsigned int)req.count);
      return ret < 0 ? ret : ERROR;
    }

  *vbuf = calloc(buffernum, sizeof(v_buffer_t));
  if (*vbuf == NULL)
    {
      printf("Out of memory for MMAP buffer metadata\n");
      return ERROR;
    }

  for (cnt = 0; cnt < req.count; cnt++)
    {
      memset(&buf, 0, sizeof(buf));
      buf.type   = type;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index  = cnt;

      ret = ioctl(fd, VIDIOC_QUERYBUF, (uintptr_t)&buf);
      if (ret < 0 || buf.length == 0)
        {
          printf("Failed to VIDIOC_QUERYBUF(MMAP) %u: errno = %d\n",
                 (unsigned int)cnt, errno);
          reset_mmap_request(fd, type);
          free_buffer(*vbuf, buffernum);
          *vbuf = NULL;
          return ret < 0 ? ret : ERROR;
        }

      (*vbuf)[cnt].length = buf.length;
      (*vbuf)[cnt].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                fd, buf.m.offset);
      if ((*vbuf)[cnt].start == (FAR uint32_t *)MAP_FAILED)
        {
          (*vbuf)[cnt].start = NULL;
          printf("Failed to mmap camera buffer %u: errno = %d\n",
                 (unsigned int)cnt, errno);
          reset_mmap_request(fd, type);
          free_buffer(*vbuf, buffernum);
          *vbuf = NULL;
          return ERROR;
        }

      (*vbuf)[cnt].mapped = true;
    }

  for (cnt = 0; cnt < req.count; cnt++)
    {
      memset(&buf, 0, sizeof(buf));
      buf.type   = type;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index  = cnt;
      buf.length = (*vbuf)[cnt].length;

      ret = ioctl(fd, VIDIOC_QBUF, (uintptr_t)&buf);
      if (ret < 0)
        {
          printf("Failed to VIDIOC_QBUF(MMAP) %u: errno = %d\n",
                 (unsigned int)cnt, errno);
          reset_mmap_request(fd, type);
          free_buffer(*vbuf, buffernum);
          *vbuf = NULL;
          return ret;
        }
    }

  ret = ioctl(fd, VIDIOC_STREAMON, (uintptr_t)&type);
  if (ret < 0)
    {
      printf("Failed to VIDIOC_STREAMON(MMAP): errno = %d\n", errno);
      reset_mmap_request(fd, type);
      free_buffer(*vbuf, buffernum);
      *vbuf = NULL;
      return ret;
    }

  return OK;
}

static uint64_t next_upload_frame_id(uint64_t timestamp_ms)
{
  uint64_t frame_id = timestamp_ms;

  if (frame_id <= g_upload_frame_id)
    {
      frame_id = g_upload_frame_id + 1;
    }

  g_upload_frame_id = frame_id;
  return frame_id;
}

static int camera_stream_is_jpeg(FAR const uint8_t *data, size_t len)
{
  size_t index;

  if (data == NULL || len < 4 || data[0] != 0xff || data[1] != 0xd8)
    {
      return ERROR;
    }

  /* Some camera/DMA paths can leave stale bytes after the current JPEG.
   * Accept the first real EOI marker and let the caller trim the stale tail;
   * checking only the last two bytes can forward a malformed payload. */
  for (index = 2; index + 1 < len; index++)
    {
      if (data[index] == 0xff && data[index + 1] == 0xd9)
        {
          return OK;
        }
    }

  return ERROR;
}

static size_t camera_stream_jpeg_length(FAR const uint8_t *data, size_t len)
{
  size_t index;

  if (camera_stream_is_jpeg(data, len) != OK)
    {
      return 0;
    }

  for (index = 2; index + 1 < len; index++)
    {
      if (data[index] == 0xff && data[index + 1] == 0xd9)
        {
          return index + 2;
        }
    }

  return 0;
}

static int camera_stream_alloc_slot(FAR struct camera_stream_ctx_s *ctx)
{
  int index;

  pthread_mutex_lock(&ctx->lock);
  for (index = 0; index < CAMERA_STREAM_QUEUE_DEPTH; index++)
    {
      if (!ctx->pool_used[index])
        {
          ctx->pool_used[index] = true;
          pthread_mutex_unlock(&ctx->lock);
          return index;
        }
    }

  ctx->dropped++;
  pthread_mutex_unlock(&ctx->lock);
  return -EAGAIN;
}

static void camera_stream_queue_release(FAR struct camera_stream_ctx_s *ctx,
                                         int index)
{
  if (index < 0 || index >= CAMERA_STREAM_QUEUE_DEPTH)
    {
      return;
    }

  pthread_mutex_lock(&ctx->lock);
  ctx->pool_used[index] = false;
  pthread_mutex_unlock(&ctx->lock);
}

static int camera_stream_queue_put(FAR struct camera_stream_ctx_s *ctx,
                                   int index)
{
  pthread_mutex_lock(&ctx->lock);

  if (ctx->queue_count >= CAMERA_STREAM_QUEUE_DEPTH)
    {
      uint8_t old = ctx->queue[ctx->queue_head];
      ctx->queue_head = (ctx->queue_head + 1) % CAMERA_STREAM_QUEUE_DEPTH;
      ctx->queue_count--;
      ctx->pool_used[old] = false;
      ctx->dropped++;
    }

  ctx->queue[ctx->queue_tail] = index;
  ctx->queue_tail = (ctx->queue_tail + 1) % CAMERA_STREAM_QUEUE_DEPTH;
  ctx->queue_count++;
  ctx->queued++;
  pthread_mutex_unlock(&ctx->lock);
  sem_post(&ctx->ready_sem);
  return OK;
}

static int camera_stream_dequeue(FAR struct camera_stream_ctx_s *ctx)
{
  int index = -1;

  pthread_mutex_lock(&ctx->lock);
  if (ctx->queue_count > 0)
    {
      index = ctx->queue[ctx->queue_head];
      ctx->queue_head = (ctx->queue_head + 1) % CAMERA_STREAM_QUEUE_DEPTH;
      ctx->queue_count--;
    }
  pthread_mutex_unlock(&ctx->lock);
  return index;
}

static FAR void *camera_stream_network_task(FAR void *arg)
{
  FAR struct camera_stream_ctx_s *ctx = arg;
  int stream_fd = -1;

  stream_fd = network_stream_open_connection();

  for (;;)
    {
      int index;
      int ret;
      struct timeval now;
      uint64_t timestamp_ms;
      uint64_t frame_id;

      while (sem_wait(&ctx->ready_sem) < 0 && errno == EINTR)
        {
        }

      index = camera_stream_dequeue(ctx);
      if (index < 0)
        {
          if (ctx->capture_done)
            {
              break;
            }

          continue;
        }

      up_invalidate_dcache((uintptr_t)ctx->pool[index],
                           (uintptr_t)ctx->pool[index] + ctx->pool_len[index]);

      gettimeofday(&now, NULL);
      timestamp_ms = (uint64_t)now.tv_sec * 1000 + now.tv_usec / 1000;
      frame_id = next_upload_frame_id(timestamp_ms);

      if (stream_fd < 0)
        {
          stream_fd = network_stream_open_connection();
        }

      ret = stream_fd < 0 ? -EIO :
            network_stream_send_frame(stream_fd,
                                      ctx->pool[index],
                                      ctx->pool_len[index],
                                      frame_id, timestamp_ms);
      if (ret < 0 && stream_fd >= 0)
        {
          /* A failed write poisons the HTTP stream.  Reconnect once and
           * resend the same queue slot before releasing its ownership. */
          close(stream_fd);
          stream_fd = network_stream_open_connection();
          ret = stream_fd < 0 ? -EIO :
                network_stream_send_frame(stream_fd,
                                          ctx->pool[index],
                                          ctx->pool_len[index],
                                          frame_id, timestamp_ms);
        }

      if (ret == OK)
        {
          ctx->sent++;
        }
      else
        {
          ctx->send_errors++;
        }

      camera_stream_queue_release(ctx, index);
    }

  if (stream_fd >= 0)
    {
      network_stream_finish_connection(stream_fd);
    }

  return NULL;
}

static int camera_text_udp_open(void)
{
  struct sockaddr_in address;
  int fd;
  int flags;

  fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    {
      printf("camera_text: UDP socket failed: %d\n", errno);
      return -1;
    }

  memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_port = htons(CAMERA_TEXT_UDP_PORT);
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, (FAR struct sockaddr *)&address, sizeof(address)) < 0)
    {
      printf("camera_text: UDP bind failed: %d\n", errno);
      close(fd);
      return -1;
    }

  flags = fcntl(fd, F_GETFL, 0);
  if (flags >= 0)
    {
      fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

  printf("camera_text: UDP listening on %d\n", CAMERA_TEXT_UDP_PORT);
  return fd;
}

static void camera_agent_send_event(int sockfd, FAR const char *event)
{
  struct sockaddr_in target;
  ssize_t sent;

  if (sockfd < 0 || event == NULL || !g_camera_text_sender_valid)
    {
      return;
    }

  memcpy(&target, &g_camera_text_sender, sizeof(target));
  target.sin_port = htons(CAMERA_HELP_EVENT_UDP_PORT);
  sent = sendto(sockfd, event, strlen(event), 0,
                (FAR struct sockaddr *)&target, sizeof(target));
  if (sent < 0)
    {
      printf("camera_agent: event send failed: %d\n", errno);
    }
}

static void camera_agent_handle_text(int sockfd, FAR const char *text)
{
  time_t now;

  if (text == NULL || text[0] == '\0')
    {
      return;
    }

  now = time(NULL);
  if (g_camera_help_pending && now >= g_camera_help_deadline)
    {
      g_camera_help_pending = false;
    }

  if (strcmp(text, "帮助") == 0)
    {
      g_camera_help_pending = true;
      g_camera_help_deadline = now + CAMERA_HELP_CONFIRM_TIMEOUT_SEC;
      camera_lcd_text_publish_pair("帮助", "是否需要帮助?");
      return;
    }

  if (g_camera_help_pending && strcmp(text, "是") == 0)
    {
      g_camera_help_pending = false;
      camera_lcd_text_publish("帮助");
      camera_agent_send_event(sockfd, "help_confirmed");
      return;
    }

  if (g_camera_help_pending)
    {
      /* Keep the confirmation prompt visible until a yes/no decision or
       * timeout.  Recognition noise must not overwrite the prompt. */
      if (strcmp(text, "不") == 0)
        {
          g_camera_help_pending = false;
          camera_lcd_text_publish("不");
        }

      return;
    }

  camera_lcd_text_publish(text);
}

static void camera_text_udp_poll(int sockfd)
{
  char packet[CAMERA_TEXT_MAX];
  char text[CAMERA_TEXT_MAX];
  struct sockaddr_in sender;
  socklen_t sender_len;
  ssize_t length;
  size_t start;
  size_t end;
  size_t copylen;
  char *label;
  char *quote;

  if (sockfd < 0)
    {
      return;
    }

  /* Drain the socket and keep only the newest text.  This is deliberately
   * non-blocking so a missing Windows sender can never pause camera capture. */
  while (1)
    {
      sender_len = sizeof(sender);
      length = recvfrom(sockfd, packet, sizeof(packet) - 1, 0,
                        (FAR struct sockaddr *)&sender, &sender_len);
      if (length <= 0)
        {
          break;
        }

      packet[length] = '\0';
      start = 0;
      while (packet[start] == ' ' || packet[start] == '\t' ||
             packet[start] == '\r' || packet[start] == '\n')
        {
          start++;
        }

      if (packet[start] == '{')
        {
          label = strstr(packet + start, "\"label\"");
          if (label != NULL)
            {
              label = strchr(label + 7, ':');
              if (label != NULL)
                {
                  label++;
                  while (*label == ' ' || *label == '\t' || *label == '"')
                    {
                      label++;
                    }

                  quote = strchr(label, '"');
                  if (quote != NULL)
                    {
                      *quote = '\0';
                    }

                  memmove(packet, label, strlen(label) + 1);
                  start = 0;
                }
              else
                {
                  continue;
                }
            }
          else
            {
              continue;
            }
        }

      end = start + strlen(packet + start);
      while (end > start &&
             (packet[end - 1] == ' ' || packet[end - 1] == '\t' ||
              packet[end - 1] == '\r' || packet[end - 1] == '\n'))
        {
          packet[--end] = '\0';
        }

      if (packet[start] == '\0')
        {
          continue;
        }

      copylen = strlen(packet + start);
      if (copylen >= sizeof(text))
        {
          copylen = sizeof(text) - 1;
        }

      memcpy(text, packet + start, copylen);
      text[copylen] = '\0';
      memcpy(&g_camera_text_sender, &sender, sizeof(sender));
      g_camera_text_sender_valid = true;
      camera_agent_handle_text(sockfd, text);
    }
}

static void camera_text_udp_close(int *sockfd)
{
  if (sockfd != NULL && *sockfd >= 0)
    {
      close(*sockfd);
      *sockfd = -1;
    }
}

static int camera_stream_init(FAR struct camera_stream_ctx_s *ctx)
{
  int index;

  memset(ctx, 0, sizeof(*ctx));
  pthread_mutex_init(&ctx->lock, NULL);
  sem_init(&ctx->ready_sem, 0, 0);

  for (index = 0; index < CAMERA_STREAM_QUEUE_DEPTH; index++)
    {
      ctx->pool[index] = memalign(32, CAMERA_STREAM_JPEG_SIZE);
      if (ctx->pool[index] == NULL)
        {
          while (--index >= 0)
            {
              free(ctx->pool[index]);
              ctx->pool[index] = NULL;
            }

          sem_destroy(&ctx->ready_sem);
          pthread_mutex_destroy(&ctx->lock);
          return ERROR;
        }

      ctx->pool_len[index] = 0;
    }

  return OK;
}

static int camera_stream_start(FAR struct camera_stream_ctx_s *ctx)
{
  pthread_attr_t attr;
  int ret;

  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, CAMERA_STREAM_STACKSIZE);
  ret = pthread_create(&ctx->network_thread, &attr,
                       camera_stream_network_task, ctx);
  pthread_attr_destroy(&attr);
  if (ret != 0)
    {
      return ERROR;
    }

  ctx->network_started = true;
  return OK;
}

static void camera_stream_stop(FAR struct camera_stream_ctx_s *ctx)
{
  if (!ctx->network_started || ctx->network_joined)
    {
      return;
    }

  ctx->capture_done = true;
  sem_post(&ctx->ready_sem);
}

static void camera_stream_join(FAR struct camera_stream_ctx_s *ctx)
{
  int index;

  if (ctx->network_started && !ctx->network_joined)
    {
      pthread_join(ctx->network_thread, NULL);
      ctx->network_joined = true;
    }

  for (index = 0; index < CAMERA_STREAM_QUEUE_DEPTH; index++)
    {
      free(ctx->pool[index]);
      ctx->pool[index] = NULL;
    }

  sem_destroy(&ctx->ready_sem);
  pthread_mutex_destroy(&ctx->lock);
}

static int camera_stream_capture_and_queue(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type,
  FAR struct camera_stream_ctx_s *ctx,
  FAR struct v_buffer *buffers,
  bool mmap_mode)
{
  int index;
  int ret;
  size_t jpeg_len;
  FAR const uint8_t *data;

  index = camera_stream_alloc_slot(ctx);
  if (index < 0)
    {
      return 0;
    }

  ret = mmap_mode ? get_camimage_mmap(fd, v4l2_buf, capture_type) :
                    get_camimage(fd, v4l2_buf, capture_type);
  if (ret != OK)
    {
      camera_stream_queue_release(ctx, index);
      return ERROR;
    }

  ctx->captured++;
  if (mmap_mode && (v4l2_buf->index >= VIDEO_BUFNUM || buffers == NULL ||
      buffers[v4l2_buf->index].start == NULL))
    {
      camera_stream_queue_release(ctx, index);
      return release_camimage(fd, v4l2_buf) == OK ? 0 : ERROR;
    }

  data = mmap_mode ? (FAR const uint8_t *)buffers[v4l2_buf->index].start :
                     (FAR const uint8_t *)v4l2_buf->m.userptr;

  jpeg_len = camera_stream_jpeg_length(data, v4l2_buf->bytesused);
  if (v4l2_buf->bytesused == 0 ||
      v4l2_buf->bytesused > CAMERA_STREAM_JPEG_SIZE ||
      jpeg_len == 0 || jpeg_len > CAMERA_STREAM_JPEG_SIZE)
    {
      ctx->bad_jpeg++;
      camera_stream_queue_release(ctx, index);
      return release_camimage(fd, v4l2_buf) == OK ? 0 : ERROR;
    }

  memcpy(ctx->pool[index],
         (FAR const void *)data, jpeg_len);
  ctx->pool_len[index] = jpeg_len;
  up_clean_dcache((uintptr_t)ctx->pool[index],
                  (uintptr_t)ctx->pool[index] + jpeg_len);

  ret = release_camimage(fd, v4l2_buf);
  if (ret != OK)
    {
      camera_stream_queue_release(ctx, index);
      return ERROR;
    }

  if (camera_stream_queue_put(ctx, index) != OK)
    {
      camera_stream_queue_release(ctx, index);
      return 0;
    }

  return 1;
}

static int frame_response_callback(FAR char **buffer, int offset,
                                   int datend, FAR int *buflen,
                                   FAR void *arg)
{
  /* The response body is only an acknowledgement.  Do not print it from the
   * network worker: serial output can block the worker and distort timing. */
  (void)buffer;
  (void)offset;
  (void)datend;
  (void)buflen;
  (void)arg;

  return 0;
}

static int upload_jpeg(FAR const uint8_t *data, size_t len)
{
  struct webclient_context ctx;
  struct timeval now;
  const char *headers[3];
  char frame_header[40];
  char timestamp_header[64];
  char response_buffer[256];
  uint64_t timestamp_ms;
  uint64_t frame_id;
  int ret;

  gettimeofday(&now, NULL);
  timestamp_ms = (uint64_t)now.tv_sec * 1000 + now.tv_usec / 1000;
  frame_id = next_upload_frame_id(timestamp_ms);

  snprintf(frame_header, sizeof(frame_header), "X-Frame-Id: %" PRIu64,
           frame_id);
  snprintf(timestamp_header, sizeof(timestamp_header),
           "X-Timestamp-Ms: %" PRIu64, timestamp_ms);
  headers[0] = "Content-Type: image/jpeg";
  headers[1] = frame_header;
  headers[2] = timestamp_header;

  memset(&ctx, 0, sizeof(ctx));
  webclient_set_defaults(&ctx);
  ctx.method = "POST";
  ctx.url = FRAME_UPLOAD_URL;
  ctx.headers = headers;
  ctx.nheaders = 3;
  ctx.buffer = response_buffer;
  ctx.buflen = sizeof(response_buffer);
  ctx.sink_callback = frame_response_callback;
  ctx.timeout_sec = CAMERA_STREAM_SEND_TIMEOUT_SEC;
  webclient_set_static_body(&ctx, data, len);

  ret = webclient_perform(&ctx);
  if (ret < 0)
    {
      printf("HTTP POST frame_id=%" PRIu64 " network_error=%d\n",
             frame_id, ret);
      return -EIO;
    }

  if (ctx.http_status < 200 || ctx.http_status >= 300)
    {
      printf("HTTP POST frame_id=%" PRIu64 " status=%u bytes=%u\n",
             frame_id, ctx.http_status, (unsigned int)len);
      return -EPROTO;
    }

  return OK;
}

static int camera_network_benchmark(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type, int frame_count)
{
  struct timeval start;
  struct timeval end;
  struct timeval delta;
  FAR const uint8_t *data;
  size_t len;
  unsigned int success = 0;
  unsigned int errors = 0;
  int index;
  int ret;

  ret = get_camimage(fd, v4l2_buf, capture_type);
  if (ret != OK)
    {
      return ret;
    }

  data = (FAR const uint8_t *)v4l2_buf->m.userptr;
  len = camera_stream_jpeg_length(data, (size_t)v4l2_buf->bytesused);
  if (len == 0)
    {
      printf("NETTEST invalid source JPEG bytes=%u\n",
             (unsigned int)v4l2_buf->bytesused);
      release_camimage(fd, v4l2_buf);
      return ERROR;
    }

  up_invalidate_dcache((uintptr_t)data, (uintptr_t)data + len);
  ret = sync_upload_frame_id();
  if (ret != OK)
    {
      release_camimage(fd, v4l2_buf);
      return ret;
    }

  printf("NETTEST fixed_jpeg_bytes=%u attempts=%d\n",
         (unsigned int)len, frame_count);
  gettimeofday(&start, NULL);

  for (index = 0; index < frame_count; index++)
    {
      ret = upload_jpeg(data, len);
      if (ret == OK)
        {
          success++;
        }
      else
        {
          errors++;
        }
    }

  gettimeofday(&end, NULL);
  timersub(&end, &start, &delta);
  printf("NETTEST stats attempts=%d success=%u errors=%u "
         "elapsed_ms=%llu fps=%.2f\n",
         frame_count,
         success,
         errors,
         (unsigned long long)(delta.tv_sec * 1000ULL +
                              delta.tv_usec / 1000ULL),
         delta.tv_sec + delta.tv_usec / 1000000.0 > 0 ?
         frame_count / (delta.tv_sec + delta.tv_usec / 1000000.0) : 0.0);

  ret = release_camimage(fd, v4l2_buf);
  return ret;
}

static int network_stream_open_connection(void)
{
  struct sockaddr_in address;
  struct timeval timeout;
  static const char request[] =
    "POST /api/v1/stream HTTP/1.1\r\n"
    "Host: 192.168.1.182:8000\r\n"
    "Content-Type: application/x-esp32-jpeg-stream\r\n"
    "Transfer-Encoding: chunked\r\n"
    "Connection: close\r\n\r\n";
  int sockfd;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0)
    {
      return -1;
    }

  timeout.tv_sec = CAMERA_STREAM_SEND_TIMEOUT_SEC;
  timeout.tv_usec = 0;
  setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_port = htons(8000);
  address.sin_addr.s_addr = inet_addr("192.168.1.182");
  if (connect(sockfd, (FAR struct sockaddr *)&address, sizeof(address)) < 0)
    {
      close(sockfd);
      return -1;
    }

  /* A blocking send can stall the whole network worker indefinitely when the
   * receiver stops consuming a long-lived stream.  Use poll-driven writes so
   * one frame has a hard upper bound. */
  {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0 || fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0)
      {
        close(sockfd);
        return -1;
      }
  }

  if (network_stream_send_all(sockfd, request, sizeof(request) - 1) != OK)
    {
      close(sockfd);
      return -1;
    }

  return sockfd;
}

static void network_stream_finish_connection(int sockfd)
{
  char response[256];
  size_t response_len = 0;

  if (network_stream_send_all(sockfd, "0\r\n\r\n", 5) != OK)
    {
      close(sockfd);
      return;
    }

  shutdown(sockfd, SHUT_WR);
  while (response_len < sizeof(response) - 1)
    {
      struct pollfd pfd;
      int poll_ret;
      ssize_t received = recv(sockfd, response + response_len,
                              sizeof(response) - 1 - response_len, 0);

      if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
          pfd.fd = sockfd;
          pfd.events = POLLIN;
          pfd.revents = 0;
          poll_ret = poll(&pfd, 1,
                          CAMERA_STREAM_SEND_TIMEOUT_SEC * 1000);
          if (poll_ret > 0 && (pfd.revents & POLLIN) != 0)
            {
              continue;
            }

          break;
        }

      if (received <= 0)
        {
          break;
        }

      response_len += received;
    }

  response[response_len] = '\0';
  if (response_len > 0 && strncmp(response, "HTTP/1.1 200", 12) != 0 &&
      strncmp(response, "HTTP/1.0 200", 12) != 0)
    {
      printf("STREAM final response: %s\n", response);
    }

  close(sockfd);
}

static int network_stream_send_all(int sockfd, FAR const void *data,
                                   size_t len)
{
  FAR const uint8_t *cursor = data;
  struct timeval start;
  struct timeval now;

  gettimeofday(&start, NULL);

  while (len > 0)
    {
      ssize_t sent = send(sockfd, cursor, len, 0);
      if (sent < 0)
        {
          if (errno == EINTR)
            {
              continue;
            }

          if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
              struct pollfd pfd;
              int elapsed_ms;
              int wait_ms;
              int poll_ret;

              gettimeofday(&now, NULL);
              elapsed_ms = (int)((now.tv_sec - start.tv_sec) * 1000 +
                                 (now.tv_usec - start.tv_usec) / 1000);
              if (elapsed_ms >= CAMERA_STREAM_SEND_TIMEOUT_SEC * 1000)
                {
                  return -ETIMEDOUT;
                }

              wait_ms = CAMERA_STREAM_SEND_TIMEOUT_SEC * 1000 - elapsed_ms;
              if (wait_ms > 250)
                {
                  wait_ms = 250;
                }

              pfd.fd = sockfd;
              pfd.events = POLLOUT;
              pfd.revents = 0;
              poll_ret = poll(&pfd, 1, wait_ms);
              if (poll_ret < 0 && errno == EINTR)
                {
                  continue;
                }

              if (poll_ret < 0 ||
                  (poll_ret > 0 &&
                   (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0))
                {
                  return -EIO;
                }

              continue;
            }

          return -EIO;
        }

      if (sent == 0)
        {
          return -EIO;
        }

      cursor += sent;
      len -= sent;
    }

  return OK;
}

static void network_stream_store_u64(FAR uint8_t *dst, uint64_t value)
{
  int index;

  for (index = 7; index >= 0; index--)
    {
      dst[index] = value & 0xff;
      value >>= 8;
    }
}

static void network_stream_store_u32(FAR uint8_t *dst, uint32_t value)
{
  int index;

  for (index = 3; index >= 0; index--)
    {
      dst[index] = value & 0xff;
      value >>= 8;
    }
}

static int network_stream_send_frame(int sockfd,
                                     FAR const uint8_t *data, size_t len,
                                     uint64_t frame_id,
                                     uint64_t timestamp_ms)
{
  uint8_t record_header[20];
  char chunk_header[16];
  int chunk_header_len;
  size_t record_len = sizeof(record_header) + len;

  network_stream_store_u64(record_header, frame_id);
  network_stream_store_u64(record_header + 8, timestamp_ms);
  network_stream_store_u32(record_header + 16, len);

  chunk_header_len = snprintf(chunk_header, sizeof(chunk_header),
                              "%x\r\n", (unsigned int)record_len);
  if (chunk_header_len <= 0 ||
      network_stream_send_all(sockfd, chunk_header,
                              (size_t)chunk_header_len) != OK ||
      network_stream_send_all(sockfd, record_header,
                              sizeof(record_header)) != OK ||
      network_stream_send_all(sockfd, data, len) != OK ||
      network_stream_send_all(sockfd, "\r\n", 2) != OK)
    {
      return -EIO;
    }

  return OK;
}

static int camera_network_stream_benchmark(
  int fd, FAR struct v4l2_buffer *v4l2_buf,
  enum v4l2_buf_type capture_type, int frame_count)
{
  struct sockaddr_in address;
  struct timeval timeout;
  struct timeval start;
  struct timeval end;
  struct timeval delta;
  FAR const uint8_t *data;
  char response[256];
  char request[] =
    "POST /api/v1/stream HTTP/1.1\r\n"
    "Host: 192.168.1.182:8000\r\n"
    "Content-Type: application/x-esp32-jpeg-stream\r\n"
    "Transfer-Encoding: chunked\r\n"
    "Connection: close\r\n\r\n";
  size_t len;
  size_t response_len = 0;
  uint64_t timestamp_ms;
  uint64_t frame_id;
  unsigned int success = 0;
  unsigned int errors = 0;
  int sockfd = -1;
  int http_status = 0;
  int index;
  int ret;

  ret = get_camimage(fd, v4l2_buf, capture_type);
  if (ret != OK)
    {
      return ret;
    }

  data = (FAR const uint8_t *)v4l2_buf->m.userptr;
  len = camera_stream_jpeg_length(data, (size_t)v4l2_buf->bytesused);
  if (len == 0)
    {
      printf("NETSTREAM invalid source JPEG bytes=%u\n",
             (unsigned int)v4l2_buf->bytesused);
      release_camimage(fd, v4l2_buf);
      return ERROR;
    }

  up_invalidate_dcache((uintptr_t)data, (uintptr_t)data + len);
  ret = sync_upload_frame_id();
  if (ret != OK)
    {
      release_camimage(fd, v4l2_buf);
      return ret;
    }

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0)
    {
      release_camimage(fd, v4l2_buf);
      return ERROR;
    }

  timeout.tv_sec = CAMERA_STREAM_SEND_TIMEOUT_SEC;
  timeout.tv_usec = 0;
  setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_port = htons(8000);
  address.sin_addr.s_addr = inet_addr("192.168.1.182");
  ret = connect(sockfd, (FAR struct sockaddr *)&address, sizeof(address));
  if (ret < 0 || network_stream_send_all(sockfd, request,
                                          sizeof(request) - 1) != OK)
    {
      close(sockfd);
      release_camimage(fd, v4l2_buf);
      return ERROR;
    }

  printf("NETSTREAM fixed_jpeg_bytes=%u attempts=%d\n",
         (unsigned int)len, frame_count);
  gettimeofday(&start, NULL);

  for (index = 0; index < frame_count; index++)
    {
      struct timeval now;

      gettimeofday(&now, NULL);
      timestamp_ms = (uint64_t)now.tv_sec * 1000 + now.tv_usec / 1000;
      frame_id = next_upload_frame_id(timestamp_ms);
      ret = network_stream_send_frame(sockfd, data, len, frame_id,
                                      timestamp_ms);
      if (ret == OK)
        {
          success++;
        }
      else
        {
          errors++;
          break;
        }
    }

  if (network_stream_send_all(sockfd, "0\r\n\r\n", 5) != OK)
    {
      errors++;
    }

  while (response_len < sizeof(response) - 1)
    {
      ssize_t received = recv(sockfd, response + response_len,
                              sizeof(response) - 1 - response_len, 0);
      if (received <= 0)
        {
          break;
        }

      response_len += received;
    }

  response[response_len] = '\0';
  if (sscanf(response, "HTTP/%*s %d", &http_status) != 1)
    {
      http_status = 0;
    }

  gettimeofday(&end, NULL);
  timersub(&end, &start, &delta);
  printf("NETSTREAM stats attempts=%d sent=%u errors=%u http_status=%d "
         "elapsed_ms=%llu fps=%.2f\n",
         frame_count,
         success,
         errors,
         http_status,
         (unsigned long long)(delta.tv_sec * 1000ULL +
                              delta.tv_usec / 1000ULL),
         delta.tv_sec + delta.tv_usec / 1000000.0 > 0 ?
         success / (delta.tv_sec + delta.tv_usec / 1000000.0) : 0.0);

  close(sockfd);
  ret = release_camimage(fd, v4l2_buf);
  return ret;
}
#endif
/****************************************************************************
 * Public Data
 ****************************************************************************/

/****************************************************************************
 * Private Functions
 ****************************************************************************/
/****************************************************************************
 * Name: preprocess_center192_to96
 *
 * Description:
 *   Crop the center 192x192 area from the 320x240 RGB565 camera frame,
 *   then downsample it to 96x96.
 ****************************************************************************/

static void preprocess_center192_to96(FAR const uint16_t *src,
                                      int src_w,
                                      int src_h,
                                      FAR uint16_t *dst)
{
  int crop_x;
  int crop_y;
  int x;
  int y;

  crop_x = (src_w - HAND_ROI_SIZE) / 2;
  crop_y = (src_h - HAND_ROI_SIZE) / 2;

  for (y = 0; y < MODEL_INPUT_SIZE; y++)
    {
      int sy = crop_y + y * 2;

      for (x = 0; x < MODEL_INPUT_SIZE; x++)
        {
          int sx = crop_x + x * 2;

          dst[y * MODEL_INPUT_SIZE + x] =
            src[sy * src_w + sx];
        }
    }
}


/****************************************************************************
 * Name: make_model_preview
 *
 * Description:
 *   Enlarge the 96x96 model image to 240x240 for LCD inspection.
 *   The destination may reuse the original camera frame buffer because
 *   the original image is no longer needed after preprocessing.
 ****************************************************************************/

static void make_model_preview(FAR const uint16_t *src,
                               FAR uint16_t *dst)
{
  int x;
  int y;

  for (y = 0; y < LCD_PREVIEW_SIZE; y++)
    {
      int sy = y * MODEL_INPUT_SIZE / LCD_PREVIEW_SIZE;

      for (x = 0; x < LCD_PREVIEW_SIZE; x++)
        {
          int sx = x * MODEL_INPUT_SIZE / LCD_PREVIEW_SIZE;

          dst[y * LCD_PREVIEW_SIZE + x] =
            src[sy * MODEL_INPUT_SIZE + sx];
        }
    }
}
/****************************************************************************
 * Name: camera_prepare()
 *
 * Description:
 *   Allocate frame buffer for camera and queue the allocated buffer
 *   into video driver.
 ****************************************************************************/

static int camera_prepare(int fd, enum v4l2_buf_type type,
                          uint32_t buf_mode, uint32_t pixformat,
                          uint16_t hsize, uint16_t vsize,
                          FAR struct v_buffer **vbuf,
                          uint8_t buffernum, int buffersize)
{
  int ret;
  int cnt;
  struct v4l2_format fmt =
  {
    0
  };

  struct v4l2_requestbuffers req =
  {
    0
  };

  struct v4l2_buffer buf =
  {
    0
  };

  /* VIDIOC_S_FMT set format */

  fmt.type                = type;
  fmt.fmt.pix.width       = hsize;
  fmt.fmt.pix.height      = vsize;
  fmt.fmt.pix.field       = V4L2_FIELD_ANY;
  fmt.fmt.pix.pixelformat = pixformat;

  ret = ioctl(fd, VIDIOC_S_FMT, (uintptr_t)&fmt);
  if (ret < 0)
    {
      printf("Failed to VIDIOC_S_FMT: errno = %d\n", errno);
      return ret;
    }

  /* VIDIOC_REQBUFS initiate user pointer I/O */

  req.type   = type;
  req.memory = V4L2_MEMORY_USERPTR;
  req.count  = buffernum;
  req.mode   = buf_mode;

  ret = ioctl(fd, VIDIOC_REQBUFS, (uintptr_t)&req);
  if (ret < 0)
    {
      printf("Failed to VIDIOC_REQBUFS: errno = %d\n", errno);
      return ret;
    }

  /* Prepare video memory to store images */

  *vbuf = malloc(sizeof(v_buffer_t) * buffernum);
  if (!(*vbuf))
    {
      printf("Out of memory for array of v_buffer_t[%d]\n", buffernum);
      return ERROR;
    }

  for (cnt = 0; cnt < buffernum; cnt++)
    {
      (*vbuf)[cnt].length = buffersize;
      (*vbuf)[cnt].mapped = false;

      /* Note:
       * VIDIOC_QBUF set buffer pointer.
       * Buffer pointer must be 32bytes aligned.
       */

      (*vbuf)[cnt].start = memalign(32, buffersize);
      if (!(*vbuf)[cnt].start)
        {
          printf("Out of memory for image buffer of %d/%d\n",
                 cnt, buffernum);

          /* Release allocated memory. */

          while (cnt--)
            {
              free((*vbuf)[cnt].start);
            }

          free(*vbuf);
          *vbuf = NULL;
          return ERROR;
        }
    }

  /* VIDIOC_QBUF enqueue buffer */

  for (cnt = 0; cnt < buffernum; cnt++)
    {
      memset(&buf, 0, sizeof(v4l2_buffer_t));
      buf.type = type;
      buf.memory = V4L2_MEMORY_USERPTR;
      buf.index = cnt;
      buf.m.userptr = (uintptr_t)(*vbuf)[cnt].start;
      buf.length = (*vbuf)[cnt].length;

      ret = ioctl(fd, VIDIOC_QBUF, (uintptr_t)&buf);
      if (ret)
        {
          printf("Fail QBUF %d\n", errno);
          free_buffer(*vbuf, buffernum);
          *vbuf = NULL;
          return ERROR;
        }
    }

  /* VIDIOC_STREAMON start stream */

  ret = ioctl(fd, VIDIOC_STREAMON, (uintptr_t)&type);
  if (ret < 0)
    {
      printf("Failed to VIDIOC_STREAMON: errno = %d\n", errno);
      free_buffer(*vbuf, buffernum);
      *vbuf = NULL;
      return ret;
    }

  return OK;
}

/****************************************************************************
 * Name: free_buffer()
 *
 * Description:
 *   All free allocated memory of v_buffer.
 ****************************************************************************/

static void free_buffer(FAR struct v_buffer *buffers, uint8_t bufnum)
{
  uint8_t cnt;
  if (buffers)
    {
      for (cnt = 0; cnt < bufnum; cnt++)
        {
        if (buffers[cnt].start)
          {
            if (buffers[cnt].mapped)
              {
                munmap(buffers[cnt].start, buffers[cnt].length);
              }
            else
              {
                free(buffers[cnt].start);
              }
          }
        }

      free(buffers);
    }
}

/****************************************************************************
 * Name: release_mmap_buffers
 ****************************************************************************/

static void release_mmap_buffers(int fd, enum v4l2_buf_type type,
                                 FAR struct v_buffer *buffers,
                                 uint8_t bufnum)
{
  struct v4l2_requestbuffers req;

  if (buffers == NULL || bufnum == 0 || !buffers[0].mapped)
    {
      return;
    }

  /* STREAMOFF is idempotent for this capture lower-half and ensures that
   * REQBUFS(0) is accepted even when the caller exits on an error path. */

  ioctl(fd, VIDIOC_STREAMOFF, (uintptr_t)&type);

  memset(&req, 0, sizeof(req));
  req.type   = type;
  req.memory = V4L2_MEMORY_MMAP;
  req.count  = 0;
  ioctl(fd, VIDIOC_REQBUFS, (uintptr_t)&req);
}

static void reset_mmap_request(int fd, enum v4l2_buf_type type)
{
  struct v4l2_requestbuffers req;

  ioctl(fd, VIDIOC_STREAMOFF, (uintptr_t)&type);

  memset(&req, 0, sizeof(req));
  req.type   = type;
  req.memory = V4L2_MEMORY_MMAP;
  req.count  = 0;
  ioctl(fd, VIDIOC_REQBUFS, (uintptr_t)&req);
}

/****************************************************************************
 * Name: parse_argument()
 *
 * Description:
 *   Parse and decode commandline arguments.
 ****************************************************************************/

static int parse_arguments(int argc, FAR char *argv[],
                           FAR int *capture_num,
                           FAR enum v4l2_buf_type *type,
                           FAR bool *upload,
                           FAR bool *stream,
                           FAR bool *stream_forever,
                           FAR bool *network_benchmark,
                           FAR bool *network_stream_benchmark)
{
  *upload = false;
  *stream = false;
  *stream_forever = false;
  *network_benchmark = false;
  *network_stream_benchmark = false;
  if (argc == 1)
    {
      *capture_num = DEFAULT_CAPTURE_NUM;
      *type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    }
  else if (argc == 2)
    {
      if (strncmp(argv[1], "-jpg", 5) == 0)
        {
          *capture_num = DEFAULT_CAPTURE_NUM;
          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
        }
      else if (strncmp(argv[1], "-upload", 8) == 0)
        {
          *capture_num = 1;
          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
          *upload = true;
        }
      else if (strncmp(argv[1], "-stream", 8) == 0)
        {
          *capture_num = 0;
          *type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
          *upload = true;
          *stream = true;
          *stream_forever = true;
        }
      else if (strncmp(argv[1], "-nettest", 8) == 0)
        {
          *capture_num = 20;
          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
          *upload = true;
          *network_benchmark = true;
        }
      else if (strncmp(argv[1], "-netstream", 10) == 0)
        {
          *capture_num = 20;
          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
          *upload = true;
          *network_stream_benchmark = true;
        }
      else
        {
          *capture_num = atoi(argv[1]);
          if (*capture_num < 0 || *capture_num > MAX_CAPTURE_NUM)
            {
              printf("Invalid capture num(%d). must be >=0 and <=%d\n",
                    *capture_num, MAX_CAPTURE_NUM);
              return ERROR;
            }

          *type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        }
    }
  else if (argc == 3)
    {
      if (strncmp(argv[1], "-jpg", 5) == 0)
        {
          *capture_num = atoi(argv[2]);
          if (*capture_num < 0 || *capture_num > MAX_CAPTURE_NUM)
            {
              printf("Invalid capture num(%d). must be >=0 and <=%d\n",
                    *capture_num, MAX_CAPTURE_NUM);
              return ERROR;
            }

          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
        }
      else if (strncmp(argv[1], "-stream", 8) == 0)
        {
          *capture_num = atoi(argv[2]);
          if (*capture_num < 1 || *capture_num > MAX_CAPTURE_NUM)
            {
              printf("Invalid stream frame count(%d). must be >=1 and <=%d\n",
                    *capture_num, MAX_CAPTURE_NUM);
              return ERROR;
            }

          *type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
          *upload = true;
          *stream = true;
        }
      else if (strncmp(argv[1], "-nettest", 8) == 0)
        {
          *capture_num = atoi(argv[2]);
          if (*capture_num < 1 || *capture_num > MAX_CAPTURE_NUM)
            {
              printf("Invalid network test frame count(%d). must be >=1 and <=%d\n",
                    *capture_num, MAX_CAPTURE_NUM);
              return ERROR;
            }

          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
          *upload = true;
          *network_benchmark = true;
        }
      else if (strncmp(argv[1], "-netstream", 10) == 0)
        {
          *capture_num = atoi(argv[2]);
          if (*capture_num < 1 || *capture_num > MAX_CAPTURE_NUM)
            {
              printf("Invalid network stream test frame count(%d). must be >=1 and <=%d\n",
                    *capture_num, MAX_CAPTURE_NUM);
              return ERROR;
            }

          *type = V4L2_BUF_TYPE_STILL_CAPTURE;
          *upload = true;
          *network_stream_benchmark = true;
        }
      else
        {
          printf("Invalid argument 1 : %s\n", argv[1]);
          return ERROR;
        }
    }
  else
    {
      printf("Too many arguments\n");
      return ERROR;
    }

  return OK;
}

/****************************************************************************
 * Name: get_camimage()
 *
 * Description:
 *   DQBUF camera frame buffer from video driver with taken picture data.
 ****************************************************************************/

static int get_camimage(int fd, FAR struct v4l2_buffer *v4l2_buf,
                        enum v4l2_buf_type buf_type)
{
  int ret;

  /* VIDIOC_DQBUF acquires captured data. */

  memset(v4l2_buf, 0, sizeof(v4l2_buffer_t));
  v4l2_buf->type = buf_type;
  v4l2_buf->memory = V4L2_MEMORY_USERPTR;

  ret = ioctl(fd, VIDIOC_DQBUF, (uintptr_t)v4l2_buf);
  if (ret)
    {
      printf("Fail DQBUF %d\n", errno);
      return ERROR;
    }

  return OK;
}

static int get_camimage_mmap(int fd, FAR struct v4l2_buffer *v4l2_buf,
                             enum v4l2_buf_type buf_type)
{
  int ret;

  /* The MMAP buffer index identifies the driver's fixed buffer heap. */

  memset(v4l2_buf, 0, sizeof(v4l2_buffer_t));
  v4l2_buf->type = buf_type;
  v4l2_buf->memory = V4L2_MEMORY_MMAP;

  ret = ioctl(fd, VIDIOC_DQBUF, (uintptr_t)v4l2_buf);
  if (ret)
    {
      printf("Fail DQBUF(MMAP) %d\n", errno);
      return ERROR;
    }

  return OK;
}

/****************************************************************************
 * Name: release_camimage()
 *
 * Description:
 *   Re-QBUF to set used frame buffer into video driver.
 ****************************************************************************/

static int release_camimage(int fd, FAR struct v4l2_buffer *v4l2_buf)
{
  int ret;

  /* VIDIOC_QBUF sets buffer pointer into video driver again. */

  ret = ioctl(fd, VIDIOC_QBUF, (uintptr_t)v4l2_buf);
  if (ret)
    {
      printf("Fail QBUF %d\n", errno);
      return ERROR;
    }

  return OK;
}

/****************************************************************************
 * Name: start_stillcapture()
 *
 * Description:
 *   Start STILL capture stream by TAKEPICT_START if buf_type is
 *   STILL_CAPTURE.
 ****************************************************************************/

static int start_stillcapture(int v_fd, enum v4l2_buf_type capture_type)
{
  int ret;

  if (capture_type == V4L2_BUF_TYPE_STILL_CAPTURE)
    {
      ret = ioctl(v_fd, VIDIOC_TAKEPICT_START, 0);
      if (ret < 0)
        {
          printf("Failed to start taking picture\n");
          return ERROR;
        }
    }

  return OK;
}

/****************************************************************************
 * Name: stop_stillcapture()
 *
 * Description:
 *   Stop STILL capture stream by TAKEPICT_STOP if buf_type is STILL_CAPTURE.
 ****************************************************************************/

static int stop_stillcapture(int v_fd, enum v4l2_buf_type capture_type)
{
  int ret;

  if (capture_type == V4L2_BUF_TYPE_STILL_CAPTURE)
    {
      ret = ioctl(v_fd, VIDIOC_TAKEPICT_STOP, false);
      if (ret < 0)
        {
          printf("Failed to stop taking picture\n");
          return ERROR;
        }
    }

  return OK;
}

/****************************************************************************
 * Name: get_imgsensor_name()
 *
 * Description:
 *   Get image sensor driver name by querying device capabilities.
 ****************************************************************************/

static FAR const char *get_imgsensor_name(int fd)
{
  static struct v4l2_capability cap;

  ioctl(fd, VIDIOC_QUERYCAP, (uintptr_t)&cap);

  return (FAR const char *)cap.driver;
}

/****************************************************************************
 * Public Functions
 ****************************************************************************/

/****************************************************************************
 * Name: main()
 *
 * Description:
 *   main routine of this example.
 ****************************************************************************/

int main(int argc, FAR char *argv[])
{
  int ret;
  int v_fd;
  int capture_num = DEFAULT_CAPTURE_NUM;
  enum v4l2_buf_type capture_type = V4L2_BUF_TYPE_STILL_CAPTURE;
  struct v4l2_buffer v4l2_buf;
  FAR const char *save_dir;
  FAR const char *sensor;
  uint16_t w;
  uint16_t h;
  int is_eternal;
  int app_state;

  struct timeval start;
  struct timeval now;
  struct timeval delta;
  struct timeval wait;
  struct timeval stream_start;
  struct timeval stream_end;
  bool upload = false;
  bool stream = false;
  bool stream_forever = false;
  bool network_benchmark = false;
  bool network_stream_benchmark = false;
  int frame_ret;
  struct camera_stream_ctx_s stream_ctx;
  bool stream_ctx_initialized = false;
  bool stream_mmap = false;
#ifdef CONFIG_NETUTILS_WEBCLIENT
  int text_udp_fd = -1;
#endif

  FAR struct v_buffer *buffers_video = NULL;
  FAR struct v_buffer *buffers_still = NULL;

  /* =====  Parse and Check arguments  ===== */

  ret = parse_arguments(argc, argv, &capture_num, &capture_type,
                        &upload, &stream, &stream_forever,
                        &network_benchmark, &network_stream_benchmark);
  if (ret != OK)
    {
      printf("usage: %s ([-jpg|-upload|-stream|-nettest|-netstream]) "
             "([capture num])\n",
             argv[0]);
      return ERROR;
    }

  /* =====  Initialization Code  ===== */

  /* Initialize NX graphics subsystem to use LCD */

#ifdef CONFIG_EXAMPLES_CAMERA_OUTPUT_LCD
  ret = nximage_initialize();
  if (ret < 0)
    {
      printf("camera_main: Failed to get NX handle: %d\n", errno);
      return ERROR;
    }
#endif

#ifdef CONFIG_EXAMPLES_CAMERA_OUTPUT_LCD
  /* The LCD is reserved for text returned by the Windows agent.  Camera
   * frames stay on the JPEG network path and are never drawn locally. */
  camera_lcd_text_init();
#endif

  /* Select storage to save image files */

  save_dir = futil_initialize();

  /* Initialize video driver to create a device file */

  ret = capture_initialize(CAMERA_DEV_PATH);
  if (ret != 0)
    {
      printf("ERROR: Failed to initialize video: errno = %d\n", errno);
      goto exit_without_cleaning_videodriver;
    }

  /* Open the device file. */

  v_fd = open(CAMERA_DEV_PATH, 0);
  if (v_fd < 0)
    {
      printf("ERROR: Failed to open video.errno = %d\n", errno);
      ret = ERROR;
      goto exit_without_cleaning_buffer;
    }

  /* Prepare for STILL_CAPTURE stream.
   *
   * The video buffer mode is V4L2_BUF_MODE_FIFO mode.
   * In this FIFO mode, if all VIDIOC_QBUFed frame buffers are captured image
   * and no additional frame buffers are VIDIOC_QBUFed, the capture stops and
   * waits for new VIDIOC_QBUFed frame buffer.
   * And when new VIDIOC_QBUF is executed, the capturing is resumed.
   *
   * Allocate frame buffers for JPEG size (512KB).
   * Set FULLHD size in ISX012 case, QUADVGA size in ISX019 case or other
   * image sensors,
   * Number of frame buffers is defined as STILL_BUFNUM(1).
   * And all allocated memorys are VIDIOC_QBUFed.
   */

  if (!stream && !network_benchmark && !network_stream_benchmark &&
      capture_num != 0)
    {
      /* Determine image size from connected image sensor name,
       * because video driver does not support VIDIOC_ENUM_FRAMESIZES
       * for now.
       */

      sensor = get_imgsensor_name(v_fd);
      if (strncmp(sensor, "ISX012", strlen("ISX012")) == 0)
        {
          w = VIDEO_HSIZE_FULLHD;
          h = VIDEO_VSIZE_FULLHD;
        }
      else if (strncmp(sensor, "ISX019", strlen("ISX019")) == 0)
        {
          w = VIDEO_HSIZE_QUADVGA;
          h = VIDEO_VSIZE_QUADVGA;
        }
      else
        {
          w = VIDEO_HSIZE_QUADVGA;
          h = VIDEO_VSIZE_QUADVGA;
        }

      ret = camera_prepare(v_fd, V4L2_BUF_TYPE_STILL_CAPTURE,
                           V4L2_BUF_MODE_FIFO, V4L2_PIX_FMT_JPEG,
                           w, h,
                           &buffers_still, STILL_BUFNUM, IMAGE_JPG_SIZE);
      if (ret != OK)
        {
          goto exit_this_app;
        }
    }

  /* Prepare for VIDEO_CAPTURE stream.
   *
   * The video buffer mode is V4L2_BUF_MODE_RING mode.
   * In this RING mode, if all VIDIOC_QBUFed frame buffers are captured image
   * and no additional frame buffers are VIDIOC_QBUFed, the capture continues
   * as the oldest image in the V4L2_BUF_QBUFed frame buffer is reused in
   * order from the captured frame buffer and a new camera image is
   * recaptured.
   *
   * Allocate freame buffers for QVGA RGB565 size (320x240x2=150KB).
   * Number of frame buffers is defined as VIDEO_BUFNUM(3).
   * And all allocated memorys are VIDIOC_QBUFed.
   */

  if (stream)
    {
      ret = camera_prepare_mmap(v_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                V4L2_BUF_MODE_RING, V4L2_PIX_FMT_JPEG,
                                VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA,
                                &buffers_video, VIDEO_BUFNUM);
      if (ret == OK)
        {
          stream_mmap = true;
        }
      else
        {
          /* Keep the stream usable on BSPs where the V4L2 mmap syscall is
           * unavailable even though the lower-half supports USERPTR. */
          printf("MMAP unavailable; falling back to fixed USERPTR buffers\n");
          ret = camera_prepare(v_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                               V4L2_BUF_MODE_RING, V4L2_PIX_FMT_JPEG,
                               VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA,
                               &buffers_video, VIDEO_BUFNUM, IMAGE_JPG_SIZE);
        }
    }
  else if (network_benchmark || network_stream_benchmark)
    {
      ret = camera_prepare(v_fd, V4L2_BUF_TYPE_STILL_CAPTURE,
                           V4L2_BUF_MODE_FIFO, V4L2_PIX_FMT_JPEG,
                           VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA,
                           &buffers_still, STILL_BUFNUM, IMAGE_JPG_SIZE);
    }
  else
    {
      ret = camera_prepare(v_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                           V4L2_BUF_MODE_RING, V4L2_PIX_FMT_RGB565,
                           VIDEO_HSIZE_QVGA, VIDEO_VSIZE_QVGA,
                           &buffers_video, VIDEO_BUFNUM, IMAGE_RGB_SIZE);
    }
  if (ret != OK)
    {
      goto exit_this_app;
    }

  if (stream)
    {
#ifdef CONFIG_NETUTILS_WEBCLIENT
      ret = sync_upload_frame_id();
      if (ret != OK)
        {
          goto exit_this_app;
        }
#endif

      ret = camera_stream_init(&stream_ctx);
      if (ret != OK)
        {
          goto exit_this_app;
      }

      stream_ctx_initialized = true;
      if (camera_stream_start(&stream_ctx) != OK)
        {
          ret = ERROR;
          goto exit_this_app;
        }

      /* Keep text reception in this process.  A second agent task/socket can
       * starve or disrupt the ESP32 network path, while this non-blocking
       * poller shares the camera loop and never waits for Windows. */
      text_udp_fd = camera_text_udp_open();
    }

  /* This application has 3 states.
   *
   * APP_STATE_BEFORE_CAPTURE:
   *    This state waits 5 seconds (defined as START_CAPTURE_TIME)
   *    with displaying preview (VIDEO_CAPTURE stream image) on LCD.
   *    After 5 seconds, state will be changed to APP_STATE_UNDER_CAPTURE.
   *
   * APP_STATE_UNDER_CAPTURE:
   *    This state will start taking picture and store the image into files.
   *    Number of taking pictures is set capture_num valiable.
   *    It can be changed by command line argument.
   *    After finishing taking pictures, the state will be changed to
   *    APP_STATE_AFTER_CAPTURE.
   *
   * APP_STATE_AFTER_CAPTURE:
   *    This state waits 10 seconds (defined as KEEP_VIDEO_TIME)
   *    with displaying preview (VIDEO_CAPTURE stream image) on LCD.
   *    After 10 seconds, this application will be finished.
   *
   * Notice:
   *    If capture_num is set '0', state will stay APP_STATE_BEFORE_CAPTURE.
   */

  app_state = APP_STATE_BEFORE_CAPTURE;

  /* Show this application behavior. */

  if (stream)
    {
      is_eternal = 1;
      app_state = APP_STATE_UNDER_CAPTURE;
      if (capture_num == 0)
        {
          printf("Start JPEG stream. Reset the board to stop.\n");
        }
      else
        {
          printf("Upload %d JPEG stream frame(s).\n", capture_num);
        }
    }
  else if (network_benchmark)
    {
      is_eternal = 0;
      app_state = APP_STATE_UNDER_CAPTURE;
      printf("Benchmark network upload with one fixed JPEG, %d attempt(s).\n",
             capture_num);
    }
  else if (network_stream_benchmark)
    {
      is_eternal = 0;
      app_state = APP_STATE_UNDER_CAPTURE;
      printf("Benchmark persistent stream with one fixed JPEG, "
             "%d attempt(s).\n", capture_num);
    }
  else if (capture_num == 0)
    {
      is_eternal = 1;
      printf("Start video this mode is eternal."
             " (Non stop, non save files.)\n");
#ifndef CONFIG_EXAMPLES_CAMERA_OUTPUT_LCD
      printf("This mode should be run with LCD display\n");
#endif
    }
  else
    {
      is_eternal = 0;
      wait.tv_sec = START_CAPTURE_TIME;
      wait.tv_usec = 0;
      if (upload)
        {
          printf("Upload %d JPEG picture(s) after %d seconds.\n",
                 capture_num, START_CAPTURE_TIME);
        }
      else
        {
          printf("Take %d pictures as %s file in %s after %d seconds.\n",
                 capture_num,
                 capture_type == V4L2_BUF_TYPE_STILL_CAPTURE ? "JPEG" : "RGB",
                 save_dir, START_CAPTURE_TIME);
        }
      printf(" After finishing taking pictures,"
             " this app will be finished after %d seconds.\n",
              KEEP_VIDEO_TIME);
    }

  gettimeofday(&start, NULL);

  /* =====  Main Loop  ===== */

  while (1)
    {
      switch (app_state)
        {
          /* BEFORE_CAPTURE and AFTER_CAPTURE is waiting for expiring the
           * time.
           * In the meantime, Capturing VIDEO image to show pre-view on LCD.
           */

          case APP_STATE_BEFORE_CAPTURE:
          case APP_STATE_AFTER_CAPTURE:
            ret = get_camimage(v_fd, &v4l2_buf, V4L2_BUF_TYPE_VIDEO_CAPTURE);
            if (ret != OK)
              {
                goto exit_this_app;
              }

#ifdef CONFIG_EXAMPLES_CAMERA_OUTPUT_LCD
              /* The ESP32-S3 camera DMA writes outside the CPU cache.  Drop
               * stale cache lines before reading the RGB565 frame. */
              up_invalidate_dcache(
                (uintptr_t)v4l2_buf.m.userptr,
                (uintptr_t)v4l2_buf.m.userptr + IMAGE_RGB_SIZE);

              /* Extract the 96x96 model input only while the dequeued frame
               * buffer is owned by this task.
               */

              preprocess_center192_to96(
                (FAR const uint16_t *)v4l2_buf.m.userptr,
                VIDEO_HSIZE_QVGA,
                VIDEO_VSIZE_QVGA,
                g_model_rgb565);

              camera_rgb565_to_rgb888(
                g_model_rgb565,
                g_model_rgb888,
                MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

              if (!g_model_rgb888_reported)
                {
                  printf("Model RGB888 ready (%u bytes), first pixel: "
                         "R=%u G=%u B=%u\n",
                         (unsigned int)sizeof(g_model_rgb888),
                         g_model_rgb888[0],
                         g_model_rgb888[1],
                         g_model_rgb888[2]);
                  g_model_rgb888_reported = true;
                }

              /* Reuse the dequeued 320x240 frame buffer for the temporary
               * 240x240 model-input preview.  The largest destination index
               * remains within the original QVGA allocation.
               */

              make_model_preview(
                g_model_rgb565,
                (FAR uint16_t *)v4l2_buf.m.userptr);

              /* Do not draw camera pixels on the LCD.  The LCD is updated
               * only by camera_lcd_text_poll() when a new agent label
               * arrives. */
              camera_lcd_text_poll();
#endif

            ret = release_camimage(v_fd, &v4l2_buf);
            if (ret != OK)
              {
                goto exit_this_app;
              }

            if (!is_eternal)
              {
                gettimeofday(&now, NULL);
                timersub(&now, &start, &delta);
                if (timercmp(&delta, &wait, > /* For checkpatch */))
                  {
                    printf("Expire time is pasted. GoTo next state.\n");
                    if (app_state == APP_STATE_BEFORE_CAPTURE)
                      {
                        app_state = APP_STATE_UNDER_CAPTURE;
                      }
                    else
                      {
                        ret = OK;
                        goto exit_this_app;
                      }
                  }
              }

            break; /* Finish APP_STATE_BEFORE_CAPTURE or APP_STATE_AFTER_CAPTURE */

          /* UNDER_CAPTURE is taking pictures until number of capture_num
           * value.
           * This state stays until finishing all pictures.
           */

          case APP_STATE_UNDER_CAPTURE:
            printf("Start capturing...\n");
            ret = OK;
            if (capture_type == V4L2_BUF_TYPE_STILL_CAPTURE)
              {
                ret = start_stillcapture(v_fd, capture_type);
                if (ret != OK)
                  {
                    goto exit_this_app;
                  }
              }

            gettimeofday(&stream_start, NULL);
            while (capture_num || stream_forever)
              {
#ifdef CONFIG_NETUTILS_WEBCLIENT
                if (network_benchmark)
                  {
                    ret = camera_network_benchmark(v_fd, &v4l2_buf,
                                                   capture_type,
                                                   capture_num);
                    if (ret != OK)
                      {
                        goto exit_this_app;
                      }

                    capture_num = 0;
                    break;
                  }

                if (network_stream_benchmark)
                  {
                    ret = camera_network_stream_benchmark(
                      v_fd, &v4l2_buf, capture_type, capture_num);
                    if (ret != OK)
                      {
                        goto exit_this_app;
                      }

                    capture_num = 0;
                    break;
                  }
#endif

                if (stream)
                  {
                    camera_text_udp_poll(text_udp_fd);
                    frame_ret = camera_stream_capture_and_queue(
                      v_fd, &v4l2_buf, capture_type, &stream_ctx,
                      buffers_video, stream_mmap);
                    if (frame_ret < 0)
                      {
                        ret = ERROR;
                        goto exit_this_app;
                      }

                    if (frame_ret > 0 && capture_num > 0)
                      {
                        capture_num--;
                      }

                    /* A full fixed pool means the network task is still
                     * draining an older JPEG.  Yield briefly so it can
                     * release a slot; otherwise a zero-rate configuration
                     * turns this loop into a CPU-burning busy spin. */
                    if (frame_ret == 0)
                      {
                        usleep(1000);
                      }
                    else if (CAMERA_STREAM_INTERVAL_MS > 0)
                      {
                        usleep(CAMERA_STREAM_INTERVAL_MS * 1000);
                      }
                    camera_lcd_text_poll();
                    continue;
                  }

                ret = get_camimage(v_fd, &v4l2_buf, capture_type);
                if (ret != OK)
                  {
                    goto exit_this_app;
                  }

                frame_ret = OK;
                if (upload)
                  {
#ifdef CONFIG_NETUTILS_WEBCLIENT
                    if (!stream)
                      {
                    if (upload && g_upload_frame_id == 0)
                      {
                        frame_ret = sync_upload_frame_id();
                      }

                    if (frame_ret == OK)
                      {
                        frame_ret = upload_jpeg(
                          (FAR const uint8_t *)v4l2_buf.m.userptr,
                          (size_t)v4l2_buf.bytesused);
                      }
                      }
#else
                    printf("JPEG upload requires CONFIG_NETUTILS_WEBCLIENT\n");
                    frame_ret = ERROR;
#endif
                  }
                else
                  {
                    futil_writeimage(
                      (FAR uint8_t *)v4l2_buf.m.userptr,
                      (size_t)v4l2_buf.bytesused,
                      capture_type == V4L2_BUF_TYPE_VIDEO_CAPTURE ?
                      "RGB" : "JPG");
                  }

                ret = release_camimage(v_fd, &v4l2_buf);
                if (ret != OK)
                  {
                    goto exit_this_app;
                  }

                if (frame_ret != OK)
                  {
                    ret = ERROR;
                    goto exit_this_app;
                  }

                if (capture_num > 0)
                  {
                    capture_num--;
                  }
              }

            ret = OK;
            ret = stop_stillcapture(v_fd, capture_type);
            if (ret != OK)
              {
                goto exit_this_app;
              }

            if (network_benchmark || network_stream_benchmark)
              {
                printf("Finished network benchmark.\n");
                goto exit_this_app;
              }

            if (stream)
              {
                ret = OK;
                /* Audio disabled: no microphone worker in this build. */
                camera_stream_stop(&stream_ctx);
                camera_stream_join(&stream_ctx);
                stream_ctx_initialized = false;
                gettimeofday(&stream_end, NULL);
                printf("STREAM stats captured=%u queued=%u sent=%u "
                       "dropped=%u bad_jpeg=%u send_errors=%u "
                       "elapsed_ms=%llu fps=%.2f\n",
                       stream_ctx.captured,
                       stream_ctx.queued,
                       stream_ctx.sent,
                       stream_ctx.dropped,
                       stream_ctx.bad_jpeg,
                       stream_ctx.send_errors,
                       (unsigned long long)
                       ((stream_end.tv_sec - stream_start.tv_sec) * 1000ULL +
                        (stream_end.tv_usec - stream_start.tv_usec) / 1000),
                       stream_ctx.sent > 1 ?
                       (double)(stream_ctx.sent - 1) /
                       (((stream_end.tv_sec - stream_start.tv_sec) * 1000000.0 +
                         stream_end.tv_usec - stream_start.tv_usec) /
                        1000000.0) : 0.0);
                printf("Finished JPEG stream.\n");
                goto exit_this_app;
              }

            app_state = APP_STATE_AFTER_CAPTURE;
            wait.tv_sec = KEEP_VIDEO_TIME;
            wait.tv_usec = 0;
            gettimeofday(&start, NULL);
            printf("Finished capturing...\n");
            break; /* Finish APP_STATE_UNDER_CAPTURE */

          default:
            printf("Unknown error is occurred.. state=%d\n", app_state);
            goto exit_this_app;
            break;
        }
    }

exit_this_app:
  if (stream_ctx_initialized)
    {
      camera_stream_stop(&stream_ctx);
      camera_stream_join(&stream_ctx);
    }

#ifdef CONFIG_NETUTILS_WEBCLIENT
  camera_text_udp_close(&text_udp_fd);
#endif

  release_mmap_buffers(v_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                       buffers_video, VIDEO_BUFNUM);

  /* Close video device file makes dequeue all buffers */

  close(v_fd);

  free_buffer(buffers_video, VIDEO_BUFNUM);
  free_buffer(buffers_still, STILL_BUFNUM);

exit_without_cleaning_buffer:
  capture_uninitialize(CAMERA_DEV_PATH);

exit_without_cleaning_videodriver:
#ifdef CONFIG_EXAMPLES_CAMERA_OUTPUT_LCD
  camera_lcd_text_close();
  nximage_finalize();
#endif
  return ret;
}
