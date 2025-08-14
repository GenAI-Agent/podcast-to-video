#!/usr/bin/env python3
"""
Find and examine the most recent SRT file from video generation
"""
import os
import glob
import tempfile

# Check common temp directories for SRT files
temp_dirs = ['/tmp', tempfile.gettempdir()]

def find_recent_srt():
    """Find the most recently created SRT file"""
    all_srt_files = []
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            # Find all SRT files in subdirectories
            pattern = os.path.join(temp_dir, '**/subtitles.srt')
            srt_files = glob.glob(pattern, recursive=True)
            all_srt_files.extend(srt_files)
    
    if not all_srt_files:
        print("No SRT files found in temp directories")
        return None
    
    # Find the most recent one
    most_recent = max(all_srt_files, key=os.path.getmtime)
    return most_recent

def examine_srt(srt_path):
    """Examine SRT file content"""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"SRT file: {srt_path}")
        print(f"Size: {len(content)} characters")
        print("\nContent:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        # Check if content looks valid
        lines = content.strip().split('\n')
        if len(lines) >= 4:
            print(f"\n✓ SRT appears valid with {len(lines)} lines")
        else:
            print(f"\n⚠ SRT may be invalid - only {len(lines)} lines")
            
        # Check for timing format
        if '-->' in content:
            print("✓ Contains timing markers")
        else:
            print("⚠ No timing markers found")
            
        return True
        
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return False

if __name__ == "__main__":
    srt_path = find_recent_srt()
    if srt_path:
        examine_srt(srt_path)
    else:
        # Let's also check if there are any temporary directories created by the video generator
        print("\nSearching for temp directories...")
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                # Look for recently created directories that might contain subtitles
                dirs = [d for d in os.listdir(temp_dir) if d.startswith('tmp') and os.path.isdir(os.path.join(temp_dir, d))]
                dirs.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)), reverse=True)
                
                for recent_dir in dirs[:5]:  # Check 5 most recent
                    full_dir = os.path.join(temp_dir, recent_dir)
                    srt_files = [f for f in os.listdir(full_dir) if f.endswith('.srt')]
                    if srt_files:
                        srt_path = os.path.join(full_dir, srt_files[0])
                        print(f"Found SRT in recent temp dir: {srt_path}")
                        examine_srt(srt_path)
                        break