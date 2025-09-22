'use client';

import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

// Get the API base URL from your environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
// This should be the path you hardcoded in your backend's main.py
const DEFAULT_VIDEO_PATH = "C:\\Users\\momen\\Downloads\\2103099-uhd_3840_2160_30fps.mp4";

interface SourceSelectorProps {
  onSourceChange: () => void;
}

export default function SourceSelector({ onSourceChange }: SourceSelectorProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  const handleSourceSelect = async (sourceType: 'default' | 'camera' | 'upload', path: string | number | null = null) => {
    setIsLoading(true);
    setStatusMessage(`Switching to ${sourceType} source...`);

    let config = {};
    if (sourceType === 'default') {
      config = { type: 'file', path: DEFAULT_VIDEO_PATH };
    } else if (sourceType === 'camera') {
      config = { type: 'camera', path: 0 }; // 0 for default webcam
    } else if (sourceType === 'upload' && path) {
      config = { type: 'file', path: path };
    } else {
      setIsLoading(false);
      return;
    }

    try {
      await axios.post(`${API_BASE_URL}/api/v1/traffic/source`, config);
      setStatusMessage(`Source switched to ${sourceType}. Refreshing feed...`);
      onSourceChange(); // This triggers the video refresh in the parent component
    } catch (error) {
      console.error('Error setting video source:', error);
      setStatusMessage('Error: Failed to switch video source.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!file) {
      setStatusMessage('Please select a file first.');
      return;
    }
    setIsLoading(true);
    setStatusMessage('Uploading video...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/traffic/upload`, formData);
      setStatusMessage('Upload successful! Setting as source...');
      // After successful upload, set this new file as the active source
      await handleSourceSelect('upload', response.data.file_path);
    } catch (error) {
      console.error('Error uploading file:', error);
      setStatusMessage('Error: File upload failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Video Source Control</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-2">
          <Button onClick={() => handleSourceSelect('default')} disabled={isLoading} className="flex-1">
            Analyze Default Video
          </Button>
          <Button onClick={() => handleSourceSelect('camera')} disabled={isLoading} className="flex-1">
            Analyze Live Camera
          </Button>
        </div>
        <div className="space-y-2">
          <Label htmlFor="video-upload">Or Upload a Video</Label>
          <div className="flex gap-2">
            <Input id="video-upload" type="file" onChange={handleFileChange} accept="video/*" disabled={isLoading} />
            <Button onClick={handleUploadAndAnalyze} disabled={isLoading || !file}>
              Upload & Analyze
            </Button>
          </div>
        </div>
        {statusMessage && <p className="text-sm text-gray-500 mt-2">{statusMessage}</p>}
      </CardContent>
    </Card>
  );
}