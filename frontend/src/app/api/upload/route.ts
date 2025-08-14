import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const topic = formData.get('topic') as string;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file received' },
        { status: 400 }
      );
    }

    if (!topic) {
      return NextResponse.json(
        { error: 'No topic selected' },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Save file to public/uploads directory
    const uploadDir = path.join(process.cwd(), 'public', 'uploads');
    const filename = `${Date.now()}-${file.name}`;
    const filepath = path.join(uploadDir, filename);

    // Create uploads directory if it doesn't exist
    const { mkdir } = await import('fs/promises');
    await mkdir(uploadDir, { recursive: true });

    await writeFile(filepath, buffer);

    return NextResponse.json({ 
      message: 'File uploaded successfully',
      filename: filename,
      size: file.size,
      topic: topic
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Failed to upload file' },
      { status: 500 }
    );
  }
}