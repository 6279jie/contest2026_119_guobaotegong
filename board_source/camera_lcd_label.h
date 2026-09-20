#ifndef __APPS_EXAMPLES_CAMERA_CAMERA_LCD_LABEL_H
#define __APPS_EXAMPLES_CAMERA_CAMERA_LCD_LABEL_H

#include <stdint.h>

void camera_lcd_draw_label(uint16_t *frame, int width, int height);
void camera_lcd_show_status(void);
int camera_lcd_text_init(void);
int camera_lcd_text_publish(const char *text);
int camera_lcd_text_publish_pair(const char *top, const char *bottom);
int camera_lcd_text_poll(void);
void camera_lcd_text_close(void);

#endif
