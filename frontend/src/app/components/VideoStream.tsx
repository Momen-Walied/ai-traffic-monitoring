'use client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// The component now accepts the stream URL as a prop
interface VideoStreamProps {
  streamUrl: string;
}

export default function VideoStream({ streamUrl }: VideoStreamProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Live Camera Feed</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Use the dynamic streamUrl prop */}
        <img
          // Add a key to force React to re-render the img tag when the URL changes
          key={streamUrl}
          src={streamUrl}
          alt="Live traffic monitoring feed"
          width={1920}
          height={1080}
          className="rounded-lg border"
        />
      </CardContent>
    </Card>
  );
}