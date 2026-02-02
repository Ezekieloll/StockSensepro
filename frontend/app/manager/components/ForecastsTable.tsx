import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/Table';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import { ChartIcon } from '@/components/ui/Icons';
import { ForecastItem } from '../page';
import React from 'react';

interface ForecastsTableProps {
  forecasts: ForecastItem[];
  forecastLoading: boolean;
  getStatusBadge: (status: string) => React.ReactNode;
  router: any;
}

const ForecastsTable: React.FC<ForecastsTableProps> = ({ forecasts, forecastLoading, getStatusBadge, router }) => (
  <Card glass className="lg:col-span-2">
    <CardHeader>
      <div className="flex items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2 text-lg">
            <ChartIcon size={18} className="text-primary" />
            ML Demand Forecasts
          </CardTitle>
          <CardDescription>7-day demand predictions from TFT+GNN model</CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success">Live</Badge>
          <Button variant="ghost" size="sm" onClick={() => router.push('/forecasts')}>View All</Button>
        </div>
      </div>
    </CardHeader>
    <CardContent>
      {forecastLoading ? (
        <div className="text-center py-8 text-muted">Loading forecasts...</div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Product</TableHead>
              <TableHead>Store</TableHead>
              <TableHead className="text-right">Stock</TableHead>
              <TableHead className="text-right">7d Forecast</TableHead>
              <TableHead className="text-right">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {forecasts.slice(0, 8).map((item) => (
              <TableRow key={`${item.sku}-${item.store_id}`} className="hover:bg-white/5">
                <TableCell>
                  <div>
                    <div className="font-medium text-sm">{item.product_name}</div>
                    <div className="text-xs text-muted font-mono">{item.sku}</div>
                  </div>
                </TableCell>
                <TableCell className="text-sm">{item.store_id}</TableCell>
                <TableCell className="text-right font-medium">{item.current_stock}</TableCell>
                <TableCell className="text-right">
                  <span className={item.seven_day_forecast > item.current_stock ? 'text-error' : 'text-foreground'}>
                    {item.seven_day_forecast.toFixed(0)}
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {getStatusBadge(item.stock_status)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </CardContent>
  </Card>
);

export default ForecastsTable;