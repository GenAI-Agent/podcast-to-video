'use client';

import { useState, useRef, useEffect } from 'react';
import dynamic from 'next/dynamic';

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000';

// Comprehensive hydration error suppression for browser extensions
if (typeof window !== 'undefined') {
  // Store original error handler
  const originalError = console.error;
  const originalWarn = console.warn;

  // Enhanced error suppression for development environment
  const shouldSuppressHydrationError = (message: string) => {
    const hydrationKeywords = [
      'A tree hydrated but some attributes',
      'cz-shortcut-listen',
      'data-new-gr-c-s-check-loaded',
      'data-gr-ext-installed',
      'data-lt-installed',
      'grammarly-extension',
      '__reactInternalInstance'
    ];

    return hydrationKeywords.some(keyword =>
      typeof message === 'string' && message.includes(keyword)
    );
  };

  // Override console.error with hydration suppression
  console.error = (...args) => {
    const message = String(args[0] || '');
    if (shouldSuppressHydrationError(message)) {
      return; // Suppress hydration errors from browser extensions
    }
    originalError.call(console, ...args);
  };

  // Override console.warn as well for React warnings
  console.warn = (...args) => {
    const message = String(args[0] || '');
    if (shouldSuppressHydrationError(message)) {
      return; // Suppress hydration warnings from browser extensions
    }
    originalWarn.call(console, ...args);
  };

  // Also suppress React's development warnings about hydration
  if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
    const originalReactError = window.console.error;
    if (originalReactError) {
      window.console.error = function (...args) {
        const message = String(args[0] || '');
        if (shouldSuppressHydrationError(message)) {
          return;
        }
        return originalReactError.apply(this, args);
      };
    }
  }
}

function UploadPageComponent() {
  const [mounted, setMounted] = useState(false);
  const [articleText, setArticleText] = useState<string>('');
  const [webLink, setWebLink] = useState<string>('');
  const [selectedAudioFile, setSelectedAudioFile] = useState<File | null>(null);
  const [selectedSrtFile, setSelectedSrtFile] = useState<File | null>(null);
  const [useGptTranscript, setUseGptTranscript] = useState(false);
  const [useBlurBackground, setUseBlurBackground] = useState(false);
  const [overlayText, setOverlayText] = useState('Demo Video');
  const [blurStrength, setBlurStrength] = useState(24);
  const [aspectRatio, setAspectRatio] = useState('9:16');
  const [backendType, setBackendType] = useState<'standard' | 'realtime'>('standard');
  const [useComfyUI, setUseComfyUI] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState<string>('');
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [progress, setProgress] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');

  // Workflow state
  const [currentStep, setCurrentStep] = useState(1);
  const [scriptGenerated, setScriptGenerated] = useState(false);
  const [audioGenerated, setAudioGenerated] = useState(false);
  const [imagesAssigned, setImagesAssigned] = useState(false);
  const [assignedImages, setAssignedImages] = useState<any[]>([]);
  const [imagePreviews, setImagePreviews] = useState<any[]>([]);
  const [imageAssignmentStats, setImageAssignmentStats] = useState<{ found: number, total: number } | null>(null);
  const [preparedImagePaths, setPreparedImagePaths] = useState<string[]>([]);
  const [preparedDescriptions, setPreparedDescriptions] = useState<string[]>([]);

  // All hooks must be called before any conditional returns
  const audioInputRef = useRef<HTMLInputElement>(null);
  const srtInputRef = useRef<HTMLInputElement>(null);

  // Ensure we're only rendering on client
  useEffect(() => {
    setMounted(true);

    // Comprehensive cleanup of browser extension attributes
    const cleanupExtensionAttributes = () => {
      const elementsToClean = [document.body, document.documentElement];

      const extensionAttributes = [
        'cz-shortcut-listen',
        'data-new-gr-c-s-check-loaded',
        'data-gr-ext-installed',
        'data-lt-installed',
        'data-grammarly-extension',
        'spellcheck',
        'data-gramm',
        'data-gramm_editor',
        'data-enable-grammarly'
      ];

      elementsToClean.forEach(element => {
        if (element) {
          extensionAttributes.forEach(attr => {
            element.removeAttribute(attr);
          });
        }
      });
    };

    // Multiple cleanup attempts to catch extensions that add attributes later
    cleanupExtensionAttributes();
    const cleanup1 = setTimeout(cleanupExtensionAttributes, 50);
    const cleanup2 = setTimeout(cleanupExtensionAttributes, 200);
    const cleanup3 = setTimeout(cleanupExtensionAttributes, 500);

    // Create a mutation observer to remove attributes as they're added
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.target) {
          const target = mutation.target as Element;
          const attrName = mutation.attributeName;

          if (attrName && (
            attrName.includes('cz-shortcut') ||
            attrName.includes('data-gr') ||
            attrName.includes('gramm') ||
            attrName.includes('data-lt')
          )) {
            target.removeAttribute(attrName);
          }
        }
      });
    });

    // Observe changes to body and html elements
    observer.observe(document.body, {
      attributes: true,
      attributeOldValue: true
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeOldValue: true
    });

    return () => {
      clearTimeout(cleanup1);
      clearTimeout(cleanup2);
      clearTimeout(cleanup3);
      observer.disconnect();
    };
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full mx-auto mb-4"></div>
          <p className="text-gray-600">Loading Video Generation Studio...</p>
        </div>
      </div>
    );
  }


  const handleAudioFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedAudioFile(file);
    }
  };

  const handleSrtFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedSrtFile(file);
    }
  };

  // Optimized: Complete Video Preparation in One Call
  const handlePrepareVideo = async () => {
    if (!articleText && !webLink) {
      setGenerationStatus('Please provide article text or web link');
      return;
    }

    setGenerating(true);
    setGenerationStatus('');
    setProgress('Preparing video content...');

    try {
      setProgress('Generating script and audio...');

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 second timeout for complete workflow

      const response = await fetch(`${API_BASE_URL}/api/prepare-video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: articleText || webLink,
          topic: 'general',
          use_gpt_transcript: useGptTranscript,
          backend_type: backendType,
          use_comfyui: useComfyUI && backendType === 'realtime'
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (response.ok) {
        // Set all data from single optimized call
        setTranscript(data.transcript);
        setAudioUrl(`${API_BASE_URL}${data.audio_url}`);
        setImagePreviews(data.image_previews || []);
        setImageAssignmentStats({
          found: data.total_images || 0,
          total: data.total_segments || 0
        });

        // Store prepared image data for consistent video generation
        setPreparedImagePaths(data.image_paths || []);
        setPreparedDescriptions(data.descriptions || []);

        // Update state flags to show completion
        setScriptGenerated(true);
        setAudioGenerated(true);
        setImagesAssigned(true);
        setCurrentStep(2); // Jump directly to video generation step (new 2-step process)

        // Legacy assigned images for compatibility
        const sentences = data.transcript.split(/[.!?]+/).filter((s: string) => s.trim());
        setAssignedImages(sentences);

        setGenerationStatus(`✅ Video preparation complete! Script generated, audio created (${data.duration?.toFixed(2)}s), ${data.total_images}/${data.total_segments} images assigned.`);
        setProgress('');

        console.log('Optimized workflow completed:', {
          transcript: data.transcript.length + ' chars',
          duration: data.duration,
          images: `${data.total_images}/${data.total_segments}`
        });
      } else {
        setGenerationStatus(`Error: ${data.error || 'Failed to prepare video'}`);
        setProgress('');
      }
    } catch (error) {
      console.error('Video preparation error:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        setGenerationStatus('Video preparation timed out. Please try again.');
      } else if (error instanceof Error && error.message.includes('Failed to fetch')) {
        setGenerationStatus('Connection error. Please check that the API server is running on port 5000.');
      } else {
        setGenerationStatus(`An error occurred: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
      setProgress('');
    } finally {
      setGenerating(false);
    }
  };

  // Step 2: Generate Final Video (Streamlined - No Regeneration)
  const handleGenerateVideo = async () => {
    if (!imagesAssigned || !transcript || !audioUrl) {
      setGenerationStatus('Please complete video preparation first');
      return;
    }

    setGenerating(true);
    setGenerationStatus('');
    setVideoUrl('');
    setProgress('Creating final video...');

    try {
      // Extract audio filename from audioUrl (e.g., "/api/download/audio_20250812_102327.mp3" -> "audio_20250812_102327.mp3")
      const audioFilename = audioUrl.split('/').pop() || '';

      if (!audioFilename) {
        throw new Error('Audio filename could not be extracted from prepared audio');
      }

      const formData = new FormData();
      formData.append('transcript', transcript); // Use prepared transcript
      formData.append('audio_filename', audioFilename); // Use prepared audio
      formData.append('topic', 'general');
      formData.append('use_blur_background', String(useBlurBackground));
      formData.append('overlay_text', overlayText);
      formData.append('blur_strength', String(blurStrength));
      formData.append('aspect_ratio', aspectRatio);
      formData.append('backend_type', backendType);

      // Include prepared image data for consistency with preview
      if (preparedImagePaths.length > 0 && preparedDescriptions.length > 0) {
        formData.append('image_paths', JSON.stringify(preparedImagePaths));
        formData.append('descriptions', JSON.stringify(preparedDescriptions));
        console.log('Including prepared image data:', {
          imagePaths: preparedImagePaths.length,
          descriptions: preparedDescriptions.length
        });
      }

      // Override with custom files if provided
      if (selectedAudioFile) {
        formData.append('custom_audio', selectedAudioFile);
      }
      if (selectedSrtFile) {
        formData.append('custom_srt', selectedSrtFile);
      }

      setProgress('Creating video from prepared content...');
      const response = await fetch(`${API_BASE_URL}/api/create-final-video`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setGenerationStatus(`✅ Final video created successfully! No content was regenerated.`);
        setVideoUrl(`${API_BASE_URL}${data.video_url}`);
        setProgress('');

        console.log('Streamlined video generation completed:', {
          transcript_chars: transcript.length,
          audio_reused: audioFilename,
          video_url: data.video_url
        });
      } else {
        setGenerationStatus(`Error: ${data.error || 'Failed to create final video'}`);
        setProgress('');
      }
    } catch (error) {
      setGenerationStatus('An error occurred during final video creation');
      console.error('Final video creation error:', error);
      setProgress('');
    } finally {
      setGenerating(false);
    }
  };

  // Reset workflow
  const handleReset = () => {
    setCurrentStep(1);
    setScriptGenerated(false);
    setAudioGenerated(false);
    setImagesAssigned(false);
    setTranscript('');
    setAudioUrl('');
    setVideoUrl('');
    setAssignedImages([]);
    setImagePreviews([]);
    setImageAssignmentStats(null);
    setGenerationStatus('');
  };

  const getStepStatus = (step: number) => {
    // For the new 2-step process: step 1 = prepare, step 2 = generate video
    // Map old currentStep (which can be 5) to new 2-step system
    const mappedCurrentStep = currentStep >= 5 ? 2 : 1;

    if (step < mappedCurrentStep) return 'completed';
    if (step === mappedCurrentStep) return 'active';
    return 'pending';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-gray-900 mb-8">
          Video Generation Studio
        </h1>

        {/* Workflow Progress */}
        <div className="bg-white shadow-xl rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            {[1, 2].map((step) => (
              <div key={step} className="flex items-center">
                <div className={`flex items-center justify-center w-12 h-12 rounded-full ${getStepStatus(step) === 'completed' ? 'bg-green-500 text-white' :
                  getStepStatus(step) === 'active' ? 'bg-blue-500 text-white' :
                    'bg-gray-300 text-gray-600'
                  }`}>
                  {getStepStatus(step) === 'completed' ? '✓' : step}
                </div>
                {step < 2 && (
                  <div className={`w-full h-1 mx-4 ${step < currentStep ? 'bg-green-500' : 'bg-gray-300'
                    }`} style={{ width: '200px' }} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between mt-2">
            <span className="text-sm font-medium">Prepare Content</span>
            <span className="text-sm font-medium">Generate Video</span>
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-500">Script + Audio + Images</span>
            <span className="text-xs text-gray-500">Final Video</span>
          </div>
        </div>

        <div className="bg-white shadow-xl rounded-lg p-8">
          <div className="space-y-6">

            {/* Step 1: Content Input and Preparation */}
            {currentStep === 1 && (
              <div className="border border-blue-500 bg-blue-50 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Step 1: Prepare Video Content
                </h2>

                {/* Content Input */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Article Text
                  </label>
                  <textarea
                    value={articleText}
                    onChange={(e) => setArticleText(e.target.value)}
                    className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter your article text here..."
                  />
                </div>

                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    OR Web Link
                  </label>
                  <input
                    type="url"
                    value={webLink}
                    onChange={(e) => setWebLink(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="https://example.com/article"
                  />
                </div>

                <div className="mb-6">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={useGptTranscript}
                      onChange={(e) => setUseGptTranscript(e.target.checked)}
                      className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="text-sm text-gray-700">Use GPT to generate optimized 1-minute script</span>
                  </label>
                </div>

                {/* Backend Selection */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Video Generation Backend
                  </label>
                  <div className="flex items-center space-x-4">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="standard"
                        checked={backendType === 'standard'}
                        onChange={(e) => setBackendType(e.target.value as 'standard' | 'realtime')}
                        className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                      />
                      <span className="text-sm text-gray-700">Standard Generator</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="realtime"
                        checked={backendType === 'realtime'}
                        onChange={(e) => setBackendType(e.target.value as 'standard' | 'realtime')}
                        className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                      />
                      <span className="text-sm text-gray-700">Realtime Generator</span>
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    {backendType === 'standard'
                      ? 'Uses video_generator.py - Standard processing with comprehensive features'
                      : 'Uses Realtime_Video_Gen.py - Optimized for faster processing'
                    }
                  </p>
                </div>

                {/* ComfyUI Option (only show when realtime is selected) */}
                {backendType === 'realtime' && (
                  <div className="border border-blue-200 rounded-lg p-4 mb-4 bg-blue-50">
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="useComfyUI"
                        checked={useComfyUI}
                        onChange={(e) => setUseComfyUI(e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="useComfyUI" className="text-sm font-medium text-gray-700">
                        🎨 Use ComfyUI for Image Generation
                      </label>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      {useComfyUI
                        ? 'Will generate custom images using ComfyUI based on your content (slower but more tailored)'
                        : 'Will use existing images from the database (faster)'
                      }
                    </p>
                  </div>
                )}

                <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-green-800">
                        Optimized Workflow
                      </h3>
                      <div className="mt-2 text-sm text-green-700">
                        This will generate your script, create audio, and assign images all in one optimized process. No more multiple audio files!
                      </div>
                    </div>
                  </div>
                </div>

                <div className="text-center">
                  <button
                    onClick={handlePrepareVideo}
                    disabled={generating || (!articleText && !webLink)}
                    className={`px-8 py-3 border border-transparent text-base font-medium rounded-md text-white transition-colors ${generating || (!articleText && !webLink)
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 cursor-pointer shadow-lg'
                      }`}
                  >
                    {generating ? 'Preparing Content...' : '🚀 Prepare Video Content'}
                  </button>
                </div>

                {/* Optional Custom Files */}
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Optional: Custom Files</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <input
                        ref={audioInputRef}
                        type="file"
                        accept=".mp3,.wav"
                        onChange={handleAudioFileSelect}
                        className="hidden"
                        id="audio-input"
                      />
                      <label
                        htmlFor="audio-input"
                        className="block w-full px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 cursor-pointer text-center"
                      >
                        {selectedAudioFile ? selectedAudioFile.name : 'Upload Custom Audio'}
                      </label>
                    </div>
                    <div>
                      <input
                        ref={srtInputRef}
                        type="file"
                        accept=".srt"
                        onChange={handleSrtFileSelect}
                        className="hidden"
                        id="srt-input"
                      />
                      <label
                        htmlFor="srt-input"
                        className="block w-full px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 cursor-pointer text-center"
                      >
                        {selectedSrtFile ? selectedSrtFile.name : 'Upload Custom Subtitles'}
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Generate Final Video */}
            {currentStep === 2 && (
              <div className="border border-blue-500 bg-blue-50 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Step 2: Generate Final Video
                </h2>

                {/* Video Options */}
                <div className="space-y-4 mb-6">
                  <div className="flex items-center space-x-6">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={useBlurBackground}
                        onChange={(e) => setUseBlurBackground(e.target.checked)}
                        className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="text-sm text-gray-700">Blur Background Effect</span>
                    </label>
                  </div>

                  {useBlurBackground && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Overlay Text
                        </label>
                        <input
                          type="text"
                          value={overlayText}
                          onChange={(e) => setOverlayText(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Blur Strength: {blurStrength}
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={blurStrength}
                          onChange={(e) => setBlurStrength(Number(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Aspect Ratio
                    </label>
                    <div className="flex space-x-4">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="9:16"
                          checked={aspectRatio === '9:16'}
                          onChange={(e) => setAspectRatio(e.target.value)}
                          className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                        />
                        <span className="text-sm text-gray-700">9:16 (Vertical)</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="16:9"
                          checked={aspectRatio === '16:9'}
                          onChange={(e) => setAspectRatio(e.target.value)}
                          className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                        />
                        <span className="text-sm text-gray-700">16:9 (Horizontal)</span>
                      </label>
                    </div>
                  </div>
                </div>

                <button
                  onClick={handleGenerateVideo}
                  disabled={generating || !imagesAssigned}
                  className={`px-6 py-3 border border-transparent text-sm font-medium rounded-md text-white transition-colors ${generating || !imagesAssigned
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                    }`}
                >
                  {generating ? 'Generating Video...' : 'Generate Final Video'}
                </button>
              </div>
            )}

            {/* Reset Button */}
            {currentStep === 2 && (
              <div className="flex justify-end">
                <button
                  onClick={handleReset}
                  className="px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  Start Over
                </button>
              </div>
            )}

            {/* Progress Indicator */}
            {progress && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <svg className="animate-spin h-5 w-5 mr-3 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span className="text-blue-800">{progress}</span>
                </div>
              </div>
            )}

            {/* Status Message */}
            {generationStatus && (
              <div className={`p-4 rounded-lg ${generationStatus.includes('successfully')
                ? 'bg-green-50 text-green-800'
                : generationStatus.includes('Error')
                  ? 'bg-red-50 text-red-800'
                  : generationStatus.includes('copied')
                    ? 'bg-blue-50 text-blue-800'
                    : 'bg-yellow-50 text-yellow-800'
                }`}>
                {generationStatus}
              </div>
            )}
          </div>
        </div>

        {/* Previews Section - Always at Bottom */}
        <div className="space-y-6">
          {/* Transcript Display */}
          {transcript && (
            <div className="bg-white shadow-xl rounded-lg p-6">
              <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-1">
                      📝 Generated Script Preview
                    </h3>
                    <p className="text-sm text-blue-600">
                      {useGptTranscript ? '✨ AI-optimized for 1-minute video' : 'Original text for audio generation'}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(transcript);
                      setGenerationStatus('Script copied to clipboard!');
                      setTimeout(() => setGenerationStatus(''), 3000);
                    }}
                    className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-sm"
                  >
                    📋 Copy Script
                  </button>
                </div>
                <div className="bg-white p-4 rounded-lg border border-blue-200 max-h-80 overflow-y-auto shadow-inner">
                  <p className="text-gray-800 whitespace-pre-wrap leading-relaxed text-sm">{transcript}</p>
                </div>
                <div className="mt-3 flex justify-between items-center text-sm">
                  <div className="text-gray-600">
                    📊 <strong>{transcript.split(/\s+/).filter(word => word.length > 0).length}</strong> words |
                    <strong> {transcript.length}</strong> characters
                  </div>
                  <div className="text-blue-600 font-medium">
                    ⏱️ Estimated reading time: ~{Math.ceil(transcript.split(/\s+/).length / 180)} min
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Audio Preview */}
          {audioUrl && (
            <div className="bg-white shadow-xl rounded-lg p-6">
              <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-1">
                      🎵 Generated Audio Preview
                    </h3>
                    <p className="text-sm text-green-600">
                      High-quality AI voice synthesis from your script
                    </p>
                  </div>
                  <a
                    href={audioUrl}
                    download
                    className="px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors shadow-sm flex items-center gap-2"
                  >
                    💾 Download MP3
                  </a>
                </div>
                <div className="bg-white p-4 rounded-lg border border-green-200 shadow-inner">
                  <audio
                    controls
                    className="w-full h-12"
                    src={audioUrl}
                    preload="metadata"
                  >
                    Your browser does not support the audio tag.
                  </audio>
                </div>
                <div className="mt-3 flex justify-between items-center text-sm">
                  <div className="text-gray-600">
                    🔊 Ready for Image Assignment
                  </div>
                  <div className="text-green-600 font-medium">
                    ✅ Audio generated successfully
                  </div>
                </div>
              </div>
            </div>
          )}


          {/* Image Previews */}
          {imagePreviews.length > 0 && (
            <div className="bg-white shadow-xl rounded-lg p-6">
              <div className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-xl">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-1">
                      🎨 Selected Images Preview
                    </h3>
                    <p className="text-sm text-purple-600">
                      Images used in your video generation
                    </p>
                  </div>
                  <div className="text-sm bg-purple-600 text-white px-3 py-1 rounded-lg">
                    {imagePreviews.filter(p => p.has_image).length} of {imagePreviews.length} images found
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-4 max-h-[600px] overflow-y-auto">
                  {imagePreviews.map((preview, index) => (
                    <div key={index} className="bg-white rounded-lg border border-purple-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                      {/* Header with index and status */}
                      <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 border-b border-purple-100">
                        <div className="flex items-center space-x-2">
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${preview.has_image ? 'bg-green-500 text-white' : 'bg-gray-400 text-white'
                            }`}>
                            {index + 1}
                          </div>
                          <span className="text-xs font-medium text-purple-700">
                            Segment {index + 1}
                          </span>
                        </div>
                        <div className="flex items-center space-x-1">
                          {preview.has_image ? (
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">✓ Found</span>
                          ) : (
                            <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">✗ Missing</span>
                          )}
                          {preview.fallback_used && (
                            <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full">Fallback</span>
                          )}
                        </div>
                      </div>

                      {/* Image preview */}
                      <div className="aspect-video bg-gray-50 relative">
                        {preview.has_image && preview.image_path ? (
                          <>
                            <img
                              src={`${API_BASE_URL}/api/image-preview/${preview.image_path}`}
                              alt={`Image ${index + 1}`}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                                const nextEl = e.currentTarget.nextElementSibling as HTMLElement;
                                if (nextEl) nextEl.style.display = 'flex';
                              }}
                            />
                            <div className="hidden w-full h-full bg-gray-100 items-center justify-center text-gray-500 text-sm">
                              <div className="text-center">
                                <div className="text-2xl mb-2">🖼️</div>
                                <div>Preview unavailable</div>
                              </div>
                            </div>
                          </>
                        ) : (
                          <div className="w-full h-full bg-gray-100 flex items-center justify-center text-gray-400">
                            <div className="text-center">
                              <div className="text-3xl mb-2">📷</div>
                              <div className="text-sm">No image found</div>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Content */}
                      <div className="p-3">
                        <p className="text-sm text-gray-800 mb-3 leading-relaxed">
                          {typeof preview.description === 'string' ? preview.description : 'No description available'}
                        </p>

                        {/* Category and tags */}
                        <div className="space-y-2">
                          {preview.category && (
                            <div className="flex items-center gap-1">
                              <span className="text-xs text-gray-500">Category:</span>
                              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded font-medium">
                                {preview.category}
                              </span>
                            </div>
                          )}
                          {preview.tags && typeof preview.tags === 'string' && (
                            <div className="flex flex-col gap-1">
                              <span className="text-xs text-gray-500">Tags:</span>
                              <div className="flex flex-wrap gap-1">
                                {preview.tags.split(',').slice(0, 3).map((tag: string, tagIndex: number) => (
                                  <span key={tagIndex} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                                    {tag.trim()}
                                  </span>
                                ))}
                                {preview.tags.split(',').length > 3 && (
                                  <span className="text-xs text-gray-400">
                                    +{preview.tags.split(',').length - 3} more
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 flex justify-between items-center text-sm">
                  <div className="text-gray-600">
                    🖼️ Images matched to script segments
                  </div>
                  <div className="text-purple-600 font-medium">
                    ✅ Ready for video generation
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Video Preview */}
          {videoUrl && (
            <div className="bg-white shadow-xl rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Generated Video</h3>
              <video
                controls
                className="w-full rounded-lg shadow-lg"
                src={videoUrl}
              >
                Your browser does not support the video tag.
              </video>
              <a
                href={videoUrl}
                download
                className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                Download Video
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Export with dynamic import to prevent SSR hydration issues
const UploadPage = dynamic(() => Promise.resolve(UploadPageComponent), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin h-8 w-8 border-b-2 border-blue-600 rounded-full mx-auto mb-4"></div>
        <p className="text-gray-600">Loading Video Generation Studio...</p>
      </div>
    </div>
  )
});

export default UploadPage;