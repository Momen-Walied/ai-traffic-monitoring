'use client'; // This page now needs client-side state

import { useState } from 'react';
import VideoStream from '@/app/components/VideoStream';
import LiveMetrics from '@/app/components/LiveMetrics';
import HistoricalChart from '@/app/components/HistoricalChart';
import ChatInterface from '@/app/components/ChatInterface';
import SourceSelector from '@/app/components/SourceSelector'; // Import the new component
import VehicleDistributionChart from '@/app/components/VehicleDistributionChart';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

export default function Home() {
  // State for the dynamic video stream URL
  const [streamUrl, setStreamUrl] = useState(`${API_BASE_URL}/api/v1/traffic/video-feed`);

  // Function to be called by SourceSelector to refresh the stream
  const handleSourceChange = () => {
    // Appending a timestamp is a common trick to bypass browser caching and
    // force the <img> tag to request the new stream from the backend.
    const newUrl = `${API_BASE_URL}/api/v1/traffic/video-feed?t=${new Date().getTime()}`;
    setStreamUrl(newUrl);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8 md:p-12 bg-gray-50">
      <div className="z-10 w-full max-w-7xl items-center justify-between">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-800 text-center mb-8">
          AI Traffic Monitoring System
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full max-w-7xl mt-6">
        {/* Main column for video and chat */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            {/* Pass the dynamic URL to the VideoStream component */}
            <VideoStream streamUrl={streamUrl} />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            {/* Add the new SourceSelector component here */}
            <SourceSelector onSourceChange={handleSourceChange} />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <ChatInterface />
          </div>
        </div>

        {/* Sidebar for metrics and charts */}
        <div className="space-y-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <LiveMetrics />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            {/* <<< ADD THE NEW CHART HERE >>> */}
            <VehicleDistributionChart />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <HistoricalChart />
          </div>
        </div>
      </div>
    </main>
  );
}