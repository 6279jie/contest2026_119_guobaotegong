$main = Get-Content -Raw "$PSScriptRoot\camera_main.c"
$lcd = Get-Content -Raw "$PSScriptRoot\camera_lcd_text.c"
$agent = Get-Content -Raw "$PSScriptRoot\text_agent_Kconfig"
$boardAgent = Get-Content -Raw "$PSScriptRoot\text_agent_main.c"
$net = Get-Content -Raw "$PSScriptRoot\nuttx_config_excerpt.txt"

if ($main -match 'camera_lcd_show_status\s*\(') {
  throw 'Camera stream still draws the startup LCD status frame.'
}

if ($main -notmatch 'CAMERA_TEXT_UDP_PORT' -or
    $main -notmatch 'camera_text_udp_poll\(text_udp_fd\)' -or
    $main -notmatch 'camera_text_udp_open\(\)') {
  throw 'Camera process does not own the non-blocking text UDP path.'
}

$helpMarkers = @('CAMERA_HELP_CONFIRM_TIMEOUT_SEC','help_confirmed','CAMERA_HELP_EVENT_UDP_PORT','camera_agent_handle_text')
foreach ($marker in $helpMarkers) {
  if ($main -notmatch [regex]::Escape($marker)) {
    throw 'Camera stream is missing the help confirmation agent/skill path.'
  }
}

if ($lcd -notmatch 'camera_lcd_text_publish\s*\(' -or
    $lcd -notmatch 'camera_lcd_text_publish_pair\s*\(' -or
    $lcd -notmatch 'camera_lcd_render_pair\s*\(') {
  throw 'LCD text renderer has no latest-wins in-process publish path.'
}

if ($main -notmatch 'camera_lcd_text_publish_pair\s*\(' -or
    $main -notmatch 'g_camera_help_pending') {
  throw 'Help prompt does not preserve the main label above the confirmation text.'
}

if ($lcd -notmatch 'camera_lcd_utf8_glyph' -or
    $lcd -notmatch '"吃"' -or
    $lcd -notmatch '"饭"' -or
    $lcd -notmatch '"谢"') {
  throw 'LCD text renderer does not contain the approved Chinese label glyph path.'
}

if ($lcd -notmatch '0xffff' -or
    $lcd -notmatch '0x7ffc' -or
    $lcd -notmatch '0x00c0' -or
    $lcd -notmatch '0x6304' -or
    $lcd -notmatch '0x7ffe' -or
    $lcd -notmatch '0x7fd8') {
  throw 'LCD Chinese glyphs do not match the generated font bitmaps.'
}

if ($lcd -notmatch '"喝"' -or
    $lcd -notmatch '"水"' -or
    $lcd -notmatch '"不"' -or
    $lcd -notmatch '"知"' -or
    $lcd -notmatch '"道"' -or
    $lcd -notmatch '"帮"' -or
    $lcd -notmatch '"助"' -or
    $lcd -notmatch '0x03fc' -or
    $lcd -notmatch '0x60fe' -or
    $lcd -notmatch '0x3fff') {
  throw 'LCD renderer is missing the requested new Chinese glyphs.'
}

foreach ($character in @('"不"','"是"','"行"','"里"','"哪"','"他"','"你"',
                         '"去"','"可"','"以"','"回"','"家"','"我"','"想"',
                         '"来"','"没"','"请"','"否"','"需"')) {
  if ($lcd -notmatch [regex]::Escape($character)) {
    throw "LCD renderer is missing model glyph $character."
  }
}

if ($lcd -notmatch '0x7fd8' -or
    $lcd -notmatch '0x1000,\s*0x11fe,\s*0x1186') {
  throw 'LCD renderer is missing the corrected 我/想 glyph bitmaps.'
}

if ($main -match 'camera_lcd_draw_label\s*\(' -or $main -match 'nximage_draw\s*\([^;]*v4l2_buf') {
  throw 'Camera preview still writes camera pixels to the LCD.'
}

if ($lcd -match 'camera_lcd_draw_label\s*\(' -or $lcd -match 'distance\s*=') {
  throw 'LCD text renderer still contains the camera/blue-ring composition.'
}

if ($lcd -notmatch 'static\s+uint16_t\s+\*?g_camera_lcd_frame') {
  throw 'LCD text renderer does not use one persistent frame buffer.'
}

if ($agent -notmatch '(?s)config\s+EXAMPLES_TEXT_AGENT_PRIORITY.*?default\s+110') {
  throw 'Text agent priority is still below the continuously-running camera task.'
}

if ($net -notmatch 'CONFIG_NET_TCP_ALLOC_CONNS=4') {
  throw 'NuttX has only one dynamically allocatable TCP connection.'
}

if ($boardAgent -notmatch 'SOCK_DGRAM' -or $boardAgent -match '\blisten\s*\(') {
  throw 'Board text agent still creates a TCP listener.'
}

Write-Output 'camera web stream / LCD text-only checks passed'
