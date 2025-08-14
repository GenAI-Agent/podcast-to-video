#!/usr/bin/env python3
"""
Test script to check SRT file generation
"""

# Create a simple test SRT file
srt_content = """1
00:00:00,000 --> 00:00:03,000
This is a test subtitle

2
00:00:03,000 --> 00:00:06,000
Second test subtitle

3
00:00:06,000 --> 00:00:09,000
Third test subtitle
"""

with open('/home/fluxmind/batch_image/test_subtitles.srt', 'w', encoding='utf-8') as f:
    f.write(srt_content)

print("Test SRT file created: /home/fluxmind/batch_image/test_subtitles.srt")
print("Content:")
print(srt_content)