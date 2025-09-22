'use client';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// Define a type for a single data point in our chart
interface ChartDataPoint {
  timestamp: string;
  vehicle_count: number;
}

export default function HistoricalChart() {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        // Construct the full URL
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/api/v1/traffic/historical-data`);
        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status}`);
        }
        const result: ChartDataPoint[] = await res.json();
        setData(result);
        setError(null);
      } catch (err) {
        console.error('Error fetching historical data:', err);
        setError('Could not load chart data.');
        setData([]); // Clear data on error
      }
    };

    fetchHistoricalData();
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Historical Data (Last 24h)</CardTitle>
      </CardHeader>
      <CardContent>
        {error ? (
          <p className="text-red-500">{error}</p>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="vehicle_count"
                stroke="#8884d8"
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

