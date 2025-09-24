'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

interface VehicleData {
  name: string;
  value: number;
  [key: string]: string | number;
}


const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AF19FF'];

export default function VehicleDistributionChart() {
  const [data, setData] = useState<VehicleData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const response = await axios.get(`${API_BASE_URL}/api/v1/traffic/vehicle-distribution`);
        setData(response.data);
      } catch (error) {
        console.error("Failed to fetch vehicle distribution data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return <Card><CardHeader><CardTitle>Vehicle Distribution (Last 24h)</CardTitle></CardHeader><CardContent><p>Loading data...</p></CardContent></Card>;
  }
  
  if (!data || data.length === 0) {
    return <Card><CardHeader><CardTitle>Vehicle Distribution (Last 24h)</CardTitle></CardHeader><CardContent><p>No vehicle data available for this period.</p></CardContent></Card>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Vehicle Distribution (Last 24h)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
              nameKey="name"
              label={({ name, percent }: { name?: string; percent?: number }) => `${name ?? ''} ${((percent ?? 0) * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}