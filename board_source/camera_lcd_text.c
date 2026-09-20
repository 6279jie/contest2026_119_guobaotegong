#include <nuttx/config.h>

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <unistd.h>

#include "camera_bkgd.h"

#define CAMERA_LCD_STATUS_SIZE 240
#define CAMERA_LCD_TEXT_QUEUE "/tmp/camera_lcd_text"
#define CAMERA_LCD_TEXT_MAX 64

static mqd_t g_camera_lcd_text_mq = (mqd_t)-1;
static uint16_t *g_camera_lcd_frame;
static char g_camera_lcd_pending[CAMERA_LCD_TEXT_MAX];
static bool g_camera_lcd_pending_valid;
static char g_camera_lcd_pending_top[CAMERA_LCD_TEXT_MAX];
static char g_camera_lcd_pending_bottom[CAMERA_LCD_TEXT_MAX];
static bool g_camera_lcd_pending_pair_valid;

static const char g_font_chars[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";

static const uint8_t g_font[38][7] =
{
  {0x0e,0x11,0x11,0x1f,0x11,0x11,0x11},
  {0x1e,0x11,0x11,0x1e,0x11,0x11,0x1e},
  {0x0f,0x10,0x10,0x10,0x10,0x10,0x0f},
  {0x1e,0x11,0x11,0x11,0x11,0x11,0x1e},
  {0x1f,0x10,0x10,0x1e,0x10,0x10,0x1f},
  {0x1f,0x10,0x10,0x1e,0x10,0x10,0x10},
  {0x0f,0x10,0x10,0x17,0x11,0x11,0x0f},
  {0x11,0x11,0x11,0x1f,0x11,0x11,0x11},
  {0x1f,0x04,0x04,0x04,0x04,0x04,0x1f},
  {0x01,0x01,0x01,0x01,0x11,0x11,0x0e},
  {0x11,0x12,0x14,0x18,0x14,0x12,0x11},
  {0x10,0x10,0x10,0x10,0x10,0x10,0x1f},
  {0x11,0x1b,0x15,0x15,0x11,0x11,0x11},
  {0x11,0x19,0x15,0x13,0x11,0x11,0x11},
  {0x0e,0x11,0x11,0x11,0x11,0x11,0x0e},
  {0x1e,0x11,0x11,0x1e,0x10,0x10,0x10},
  {0x0e,0x11,0x11,0x11,0x15,0x12,0x0d},
  {0x1e,0x11,0x11,0x1e,0x14,0x12,0x11},
  {0x0f,0x10,0x10,0x0e,0x01,0x01,0x1e},
  {0x1f,0x04,0x04,0x04,0x04,0x04,0x04},
  {0x11,0x11,0x11,0x11,0x11,0x11,0x0e},
  {0x11,0x11,0x11,0x11,0x11,0x0a,0x04},
  {0x11,0x11,0x11,0x15,0x15,0x15,0x0a},
  {0x11,0x11,0x0a,0x04,0x0a,0x11,0x11},
  {0x11,0x11,0x0a,0x04,0x04,0x04,0x04},
  {0x1f,0x01,0x02,0x04,0x08,0x10,0x1f},
  {0x0e,0x11,0x13,0x15,0x19,0x11,0x0e},
  {0x04,0x0c,0x04,0x04,0x04,0x04,0x0e},
  {0x0e,0x11,0x01,0x02,0x04,0x08,0x1f},
  {0x1e,0x01,0x01,0x0e,0x01,0x01,0x1e},
  {0x02,0x06,0x0a,0x12,0x1f,0x02,0x02},
  {0x1f,0x10,0x10,0x1e,0x01,0x01,0x1e},
  {0x0e,0x10,0x10,0x1e,0x11,0x11,0x0e},
  {0x1f,0x01,0x02,0x04,0x08,0x08,0x08},
  {0x0e,0x11,0x11,0x0e,0x11,0x11,0x0e},
  {0x0e,0x11,0x11,0x0f,0x01,0x01,0x0e},
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  {0x1f,0x01,0x02,0x04,0x08,0x10,0x1f}
};

/* The board only needs the labels currently produced by the Windows model.
 * Keep these as compact 16x16 monochrome glyphs instead of linking a full
 * Chinese font.  They are rendered at 2x scale on the 240x240 LCD. */
struct camera_lcd_chinese_glyph_s
{
  const char *utf8;
  uint16_t rows[16];
};

static const struct camera_lcd_chinese_glyph_s g_chinese_glyphs[] =
{
  {"否", {0x0000, 0x0000, 0x7ffe, 0x0180, 0x03e0, 0x05b0, 0x199c, 0x7186,
          0xc003, 0x3ffc, 0x300c, 0x300c, 0x300c, 0x300c, 0x3ffc, 0x300c}},
  {"需", {0x0000, 0x0000, 0x7ffc, 0x0180, 0x7ffe, 0x4182, 0x5ffa, 0x0180,
          0x1ff8, 0x0000, 0xffff, 0x0100, 0x3ffc, 0x2664, 0x2664, 0x267c}},
  {"有", {0x0000, 0x0200, 0x0600, 0xffff, 0x0c00, 0x0c00, 0x1ffc, 0x380c,
          0x7ffc, 0xd80c, 0x180c, 0x1ffc, 0x180c, 0x180c, 0x187c, 0x0000}},
  {"要", {0x0000, 0x0000, 0x7ffe, 0x0660, 0x7ffe, 0x6666, 0x6666, 0x7ffe,
          0x6606, 0x0600, 0xffff, 0x0830, 0x1e60, 0x07e0, 0x1ffc, 0x780e}},
  {"吃", {0x0000, 0x00c0, 0x00c0, 0x7d80, 0x4dff, 0x4f80, 0x4f00, 0x4ffc,
          0x4c1c, 0x4c30, 0x4c63, 0x7cc3, 0x4d83, 0x4d03, 0x01fe, 0x0000}},
  {"饭", {0x0000, 0x200e, 0x23fe, 0x7f00, 0x4f00, 0xcb00, 0xbbfe, 0x23c6,
          0x2346, 0x2344, 0x236c, 0x2f38, 0x3a38, 0x367c, 0x24c6, 0x0d83}},
  {"谢", {0x0000, 0x6304, 0x6304, 0x3fc4, 0x3cc4, 0x0fff, 0x0cc4, 0xeff4,
          0x6cf4, 0x6cd4, 0x6cdc, 0x6fc4, 0x79c4, 0x73c4, 0x6ec4, 0x1bdc}},
  {"喝", {0x0000, 0x03fc, 0x7b04, 0x4bfc, 0x4b04, 0x4bfc, 0x4900, 0x4bfe,
          0x4e22, 0x4f72, 0x7b9a, 0x4b02, 0x43fe, 0x0006, 0x003c, 0x0000}},
  {"水", {0x0000, 0x0180, 0x0180, 0x0184, 0x018c, 0xfdd8, 0x0df0, 0x0de0,
          0x09b0, 0x19b0, 0x3198, 0x318c, 0x6186, 0x4183, 0x0f00, 0x0000}},
  {"不", {0x0000, 0x0000, 0xffff, 0x01c0, 0x0180, 0x0380, 0x0780, 0x0da0,
          0x19b8, 0x318e, 0x6183, 0xc180, 0x0180, 0x0180, 0x0180, 0x0180}},
  {"知", {0x2000, 0x2000, 0x60fe, 0x7fc2, 0x58c2, 0xd8c2, 0xd8c2, 0x18c2,
          0xffc2, 0x18c2, 0x18c2, 0x1cc2, 0x36c2, 0x23fe, 0x63c2, 0xc0c2}},
  {"道", {0x0000, 0x4108, 0x6198, 0x3fff, 0x3040, 0x0040, 0x07fe, 0xf606,
          0x37fe, 0x3606, 0x37fe, 0x3606, 0x37fe, 0x3606, 0xfc00, 0xcfff}},
  {"帮", {0x0000, 0x0800, 0x087e, 0x7fe6, 0x086c, 0x7f66, 0x0862, 0xfffe,
          0x1060, 0x7180, 0x7ffc, 0x118c, 0x118c, 0x118c, 0x11b8, 0x0180}},
  {"助", {0x0000, 0x0000, 0x7e20, 0x6220, 0x62fe, 0x7e66, 0x6266, 0x6266,
          0x7e66, 0x6266, 0x6266, 0x6246, 0x7fc6, 0xf9c6, 0x01bc, 0x0100}},
  {"是", {0x0000, 0x0000, 0x1ffc, 0x100c, 0x1ffc, 0x100c, 0x1ffc, 0x100c,
          0x0000, 0xffff, 0x0180, 0x19fe, 0x1180, 0x3d80, 0x67ff, 0xc000}},
  {"行", {0x0000, 0x0800, 0x19fe, 0x3000, 0x6000, 0xcc00, 0x0800, 0x1bff,
          0x3018, 0x7018, 0xf018, 0x9018, 0x1018, 0x1018, 0x10f8, 0x10f0}},
  {"里", {0x0000, 0x0000, 0x3ffc, 0x2184, 0x2184, 0x3ffc, 0x2184, 0x2184,
          0x3ffc, 0x2184, 0x0180, 0x7ffe, 0x0180, 0x0180, 0xffff, 0x0000}},
  {"哪", {0x0000, 0x0000, 0x07de, 0x7ad2, 0x5ad6, 0x5fd6, 0x5ad4, 0x5adc,
          0x5ad6, 0x5fd2, 0x5ed3, 0x7ed3, 0x44d3, 0x0cde, 0x1f90, 0x1010}},
  {"他", {0x0000, 0x1860, 0x1860, 0x1260, 0x327e, 0x77fe, 0x7f66, 0xf266,
          0xb266, 0x327e, 0x3260, 0x3263, 0x3203, 0x3203, 0x33fe, 0x3000}},
  {"你", {0x0000, 0x0980, 0x1980, 0x19ff, 0x3303, 0x3626, 0x7626, 0x7020,
          0xf120, 0xb32c, 0x3326, 0x3226, 0x3623, 0x3420, 0x3060, 0x31e0}},
  {"去", {0x0000, 0x0180, 0x0180, 0x0180, 0x7ffe, 0x0180, 0x0180, 0x0180,
          0xffff, 0x0300, 0x0620, 0x0c30, 0x0c18, 0x19fc, 0x3fc4, 0x2006}},
  {"可", {0x0000, 0x0000, 0xffff, 0x0018, 0x0008, 0x3f88, 0x3188, 0x2188,
          0x2188, 0x2188, 0x2188, 0x3f88, 0x3008, 0x00f8, 0x00f8, 0x0000}},
  {"以", {0x0000, 0x020c, 0x630c, 0x630c, 0x6188, 0x6188, 0x6008, 0x6008,
          0x6018, 0x6018, 0x6218, 0x6638, 0x7c6c, 0x70c6, 0x6387, 0x0303}},
  {"回", {0x0000, 0x0000, 0x7ffe, 0x6006, 0x6006, 0x6ff6, 0x6c36, 0x6c36,
          0x6c36, 0x6c36, 0x6ff6, 0x6c36, 0x6006, 0x6006, 0x7ffe, 0x6006}},
  {"家", {0x0000, 0x0180, 0x0180, 0x7ffe, 0x4002, 0x3ffc, 0x0700, 0x1f00,
          0x730c, 0x07f8, 0x38f0, 0x61d8, 0x06ce, 0x3cc7, 0x6780, 0x0000}},
  {"我", {0x0000, 0x7fd8, 0x7e4c, 0x1846, 0x1840, 0xffff, 0x1860, 0x1866,
          0x1f6c, 0xfe78, 0xf870, 0x1873, 0x19f3, 0x1b1b, 0x780e, 0x0000}},
  {"想", {0x0000, 0x1000, 0x11fe, 0x1186, 0xfffe, 0x3d86, 0x3dfe, 0x5786,
          0xd186, 0x91fe, 0x0000, 0x6d10, 0x6c96, 0x4c33, 0xcff0, 0x0000}},
  {"来", {0x0000, 0x0180, 0x0180, 0x7ffe, 0x0180, 0x1998, 0x0db0, 0x0180,
          0xffff, 0x03c0, 0x07c0, 0x0db0, 0x1998, 0x718e, 0xc183, 0x0180}},
  {"没", {0x0000, 0x6000, 0x71fc, 0x110c, 0x030c, 0xc70f, 0x6600, 0x3000,
          0x07fc, 0x330c, 0x3198, 0x60f0, 0x60f0, 0x67fc, 0x4e0f, 0x0000}},
  {"请", {0x0000, 0x6060, 0x37ff, 0x1060, 0x07fe, 0x0060, 0xffff, 0x3000,
  0x33fe, 0x3206, 0x33fe, 0x3ffe, 0x3e06, 0x3206, 0x223c, 0x0000}}
};

static const uint16_t *camera_lcd_utf8_glyph(const char *value,
                                              size_t *consumed)
{
  unsigned int index;

  for (index = 0; index < sizeof(g_chinese_glyphs) /
                         sizeof(g_chinese_glyphs[0]); index++)
    {
      if (strncmp(value, g_chinese_glyphs[index].utf8, 3) == 0)
        {
          *consumed = 3;
          return g_chinese_glyphs[index].rows;
        }
    }

  return NULL;
}

static void camera_lcd_draw_chinese_text_at(uint16_t *frame, const char *text,
                                            int origin_y)
{
  const uint16_t *glyphs[8];
  const char *cursor = text;
  int count = 0;
  int origin_x;
  int index;
  int row;
  int col;
  int sx;
  int sy;
  size_t consumed;

  while (*cursor != '\0' && count < 8)
    {
      glyphs[count] = camera_lcd_utf8_glyph(cursor, &consumed);
      if (glyphs[count] == NULL)
        {
          /* Keep rendering the characters that are available even if a
           * future Windows label contains a glyph not yet in this compact
           * table.  The old all-or-nothing return made an entire word vanish
           * when just one character was missing. */
          if ((unsigned char)*cursor >= 0x80)
            {
              cursor += 3;
            }
          else
            {
              cursor++;
            }

          continue;
        }

      cursor += consumed;
      count++;
    }

  if (count == 0 || *cursor != '\0')
    {
      return;
    }

  origin_x = (CAMERA_LCD_STATUS_SIZE - count * 32 - (count - 1) * 6) / 2;
  for (index = 0; index < count; index++)
    {
      for (row = 0; row < 16; row++)
        {
          for (col = 0; col < 16; col++)
            {
              if ((glyphs[index][row] & (1 << (15 - col))) == 0)
                {
                  continue;
                }

              for (sy = 0; sy < 2; sy++)
                {
                  for (sx = 0; sx < 2; sx++)
                    {
                      int px = origin_x + index * 38 + col * 2 + sx;
                      int py = origin_y + row * 2 + sy;
                      if (px >= 0 && px < CAMERA_LCD_STATUS_SIZE &&
                          py >= 0 && py < CAMERA_LCD_STATUS_SIZE)
                        {
                          frame[py * CAMERA_LCD_STATUS_SIZE + px] = 0xffff;
                        }
                    }
                }
            }
        }
    }
}

static void camera_lcd_draw_chinese_text(uint16_t *frame, const char *text)
{
  camera_lcd_draw_chinese_text_at(frame, text, 88);
}

static const uint8_t *camera_lcd_glyph(char value)
{
  const char *found;

  if (value >= 'a' && value <= 'z')
    {
      value = value - 'a' + 'A';
    }

  found = strchr(g_font_chars, value);
  return found == NULL ? NULL : g_font[(int)(found - g_font_chars)];
}

static void camera_lcd_draw_text(uint16_t *frame, const char *text)
{
  const int scale = 3;
  const int glyph_width = 5 * scale;
  const int glyph_gap = scale;
  const int glyph_height = 7 * scale;
  const int max_chars = 12;
  const char *cursor;
  int count = 0;
  int total_width;
  int origin_x;
  int origin_y;
  int index;
  int row;
  int col;
  int sx;
  int sy;

  if ((unsigned char)text[0] >= 0x80)
    {
      camera_lcd_draw_chinese_text(frame, text);
      return;
    }

  cursor = text;
  while (*cursor != '\0' && count < max_chars)
    {
      count++;
      cursor++;
    }

  if (count == 0)
    {
      return;
    }

  total_width = count * glyph_width + (count - 1) * glyph_gap;
  origin_x = (CAMERA_LCD_STATUS_SIZE - total_width) / 2;
  origin_y = (CAMERA_LCD_STATUS_SIZE - glyph_height) / 2 + 54;
  for (index = 0; index < count; index++)
    {
      const uint8_t *glyph = camera_lcd_glyph(text[index]);
      if (glyph == NULL)
        {
          continue;
        }

      for (row = 0; row < 7; row++)
        {
          for (col = 0; col < 5; col++)
            {
              if ((glyph[row] & (1 << (4 - col))) == 0)
                {
                  continue;
                }

              for (sy = 0; sy < scale; sy++)
                {
                  for (sx = 0; sx < scale; sx++)
                    {
                      int px = origin_x + index * (glyph_width + glyph_gap) +
                               col * scale + sx;
                      int py = origin_y + row * scale + sy;
                      if (px >= 0 && px < CAMERA_LCD_STATUS_SIZE &&
                          py >= 0 && py < CAMERA_LCD_STATUS_SIZE)
                        {
                          frame[py * CAMERA_LCD_STATUS_SIZE + px] = 0xffff;
                        }
                    }
                }
            }
        }
    }
}

static void camera_lcd_render_text(const char *text)
{
  if (g_camera_lcd_frame == NULL)
    {
      printf("camera_lcd: text frame is not initialized\\n");
      return;
    }

  /* Only text is rendered.  The persistent buffer avoids repeated heap
   * allocation and the camera frame is never copied to this display path. */
  memset(g_camera_lcd_frame, 0,
         CAMERA_LCD_STATUS_SIZE * CAMERA_LCD_STATUS_SIZE *
         sizeof(*g_camera_lcd_frame));
  camera_lcd_draw_text(g_camera_lcd_frame, text);
  nximage_draw(g_camera_lcd_frame, CAMERA_LCD_STATUS_SIZE,
               CAMERA_LCD_STATUS_SIZE);
}

static void camera_lcd_render_pair(const char *top, const char *bottom)
{
  if (g_camera_lcd_frame == NULL)
    {
      printf("camera_lcd: pair frame is not initialized\\n");
      return;
    }

  memset(g_camera_lcd_frame, 0,
         CAMERA_LCD_STATUS_SIZE * CAMERA_LCD_STATUS_SIZE *
         sizeof(*g_camera_lcd_frame));
  camera_lcd_draw_chinese_text_at(g_camera_lcd_frame, top, 42);
  camera_lcd_draw_chinese_text_at(g_camera_lcd_frame, bottom, 148);
  nximage_draw(g_camera_lcd_frame, CAMERA_LCD_STATUS_SIZE,
               CAMERA_LCD_STATUS_SIZE);
}

int camera_lcd_text_init(void)
{
  struct mq_attr attributes;

  memset(&attributes, 0, sizeof(attributes));
  attributes.mq_maxmsg = 1;
  attributes.mq_msgsize = CAMERA_LCD_TEXT_MAX;
  g_camera_lcd_text_mq = mq_open(CAMERA_LCD_TEXT_QUEUE,
                                 O_RDONLY | O_CREAT | O_NONBLOCK,
                                 0644, &attributes);
  if (g_camera_lcd_text_mq == (mqd_t)-1)
    {
      printf("camera_lcd: text queue open failed: %d\\n", errno);
      return -errno;
    }

  g_camera_lcd_frame = (uint16_t *)malloc(CAMERA_LCD_STATUS_SIZE *
                                          CAMERA_LCD_STATUS_SIZE *
                                          sizeof(*g_camera_lcd_frame));
  if (g_camera_lcd_frame == NULL)
    {
      mq_close(g_camera_lcd_text_mq);
      g_camera_lcd_text_mq = (mqd_t)-1;
      printf("camera_lcd: text frame allocation failed\\n");
      return -ENOMEM;
    }

  memset(g_camera_lcd_frame, 0,
         CAMERA_LCD_STATUS_SIZE * CAMERA_LCD_STATUS_SIZE *
         sizeof(*g_camera_lcd_frame));
  g_camera_lcd_pending[0] = '\0';
  g_camera_lcd_pending_valid = false;
  g_camera_lcd_pending_top[0] = '\0';
  g_camera_lcd_pending_bottom[0] = '\0';
  g_camera_lcd_pending_pair_valid = false;

  return 0;
}

int camera_lcd_text_publish(const char *text)
{
  size_t length;

  if (text == NULL || g_camera_lcd_frame == NULL)
    {
      return -EINVAL;
    }

  length = strnlen(text, sizeof(g_camera_lcd_pending) - 1);
  memcpy(g_camera_lcd_pending, text, length);
  g_camera_lcd_pending[length] = '\0';
  g_camera_lcd_pending_valid = true;
  return 0;
}

int camera_lcd_text_publish_pair(const char *top, const char *bottom)
{
  size_t top_length;
  size_t bottom_length;

  if (top == NULL || bottom == NULL || g_camera_lcd_frame == NULL)
    {
      return -EINVAL;
    }

  top_length = strnlen(top, sizeof(g_camera_lcd_pending_top) - 1);
  memcpy(g_camera_lcd_pending_top, top, top_length);
  g_camera_lcd_pending_top[top_length] = '\0';

  bottom_length = strnlen(bottom, sizeof(g_camera_lcd_pending_bottom) - 1);
  memcpy(g_camera_lcd_pending_bottom, bottom, bottom_length);
  g_camera_lcd_pending_bottom[bottom_length] = '\0';
  g_camera_lcd_pending_pair_valid = true;
  return 0;
}

int camera_lcd_text_poll(void)
{
  char received[CAMERA_LCD_TEXT_MAX];
  char latest[CAMERA_LCD_TEXT_MAX];
  ssize_t length;
  bool changed = false;

  if (g_camera_lcd_pending_pair_valid)
    {
      camera_lcd_render_pair(g_camera_lcd_pending_top,
                             g_camera_lcd_pending_bottom);
      g_camera_lcd_pending_pair_valid = false;
      g_camera_lcd_pending_valid = false;
      while (g_camera_lcd_text_mq != (mqd_t)-1 &&
             mq_receive(g_camera_lcd_text_mq, received, sizeof(received),
                        NULL) >= 0)
        {
        }
      return 1;
    }

  if (g_camera_lcd_pending_valid)
    {
      memcpy(latest, g_camera_lcd_pending, sizeof(latest));
      g_camera_lcd_pending_valid = false;
      changed = true;
    }

  if (g_camera_lcd_text_mq == (mqd_t)-1)
    {
      if (changed)
        {
          camera_lcd_render_text(latest);
          return 1;
        }

      return 0;
    }

  while ((length = mq_receive(g_camera_lcd_text_mq, received,
                              sizeof(received), NULL)) >= 0)
    {
      if (length >= (ssize_t)sizeof(received))
        {
          length = sizeof(received) - 1;
        }

      memcpy(latest, received, length);
      latest[length] = '\0';
      changed = true;
    }

  if (changed)
    {
      camera_lcd_render_text(latest);
      return 1;
    }

  return 0;
}

void camera_lcd_text_close(void)
{
  if (g_camera_lcd_text_mq != (mqd_t)-1)
    {
      mq_close(g_camera_lcd_text_mq);
      g_camera_lcd_text_mq = (mqd_t)-1;
    }

  free(g_camera_lcd_frame);
  g_camera_lcd_frame = NULL;
}
