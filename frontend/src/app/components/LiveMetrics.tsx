'use client';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Define a type for our metrics data for type safety
interface TrafficMetrics {
  vehicle_count: number | string;
  density: string;
}

export default function LiveMetrics() {
  const [metrics, setMetrics] = useState<TrafficMetrics>({
    vehicle_count: '--',
    density: '--',
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        // Construct the full URL
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/api/v1/traffic/latest-metrics`);
        if (!res.ok) {
          // If response is not OK, don't try to parse JSON
          console.error("Failed to fetch metrics, status:", res.status);
          return;
        }
        const data: TrafficMetrics = await res.json();
        setMetrics(data);
      } catch (error) {
        console.error('Error fetching live metrics:', error);
      }
    };

    const intervalId = setInterval(fetchMetrics, 5000); // Fetch every 5 seconds

    // Cleanup function to clear the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Real-Time Data</CardTitle>
      </CardHeader>
      <CardContent className="text-2xl font-bold">
        <div className="flex justify-between items-center">
          <span>Vehicles/Min:</span>
          <span>{metrics.vehicle_count}</span>
        </div>
        <div className="flex justify-between items-center mt-4">
          <span>Density:</span>
          <span className="capitalize">{metrics.density}</span>
        </div>
      </CardContent>
    </Card>
  );
}

