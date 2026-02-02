import Card, { CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { BellIcon, CheckIcon } from '@/components/ui/Icons';
import { ForecastAlert } from '../page';
import React from 'react';

interface AlertsPanelProps {
  alerts: ForecastAlert[];
  forecastLoading: boolean;
  getSeverityColor: (severity: string) => string;
}

const AlertsPanel: React.FC<AlertsPanelProps> = ({ alerts, forecastLoading, getSeverityColor }) => (
  <Card glass className="lg:col-span-1">
    <CardHeader>
      <div className="flex items-center justify-between">
        <CardTitle className="flex items-center gap-2 text-lg">
          <BellIcon size={18} className="text-error" />
          ML-Generated Alerts
          <span className="ml-1 px-2 py-0.5 bg-error/20 text-error text-xs rounded-full">{alerts.length}</span>
        </CardTitle>
      </div>
    </CardHeader>
    <CardContent>
      {forecastLoading ? (
        <div className="text-center py-8 text-muted">Loading alerts...</div>
      ) : alerts.length === 0 ? (
        <div className="text-center py-8 text-muted">
          <CheckIcon size={32} className="mx-auto mb-2 text-success" />
          <p>No alerts - all stock levels healthy!</p>
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.slice(0, 5).map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border-l-4 ${getSeverityColor(alert.severity)}`}
            >
              <p className="text-sm font-medium">{alert.message}</p>
              <p className="text-xs text-muted mt-1">
                Store: {alert.store_id} • Stock: {alert.current_stock} • Forecast: {alert.predicted_demand.toFixed(0)}
              </p>
            </div>
          ))}
        </div>
      )}
      <Button variant="ghost" className="w-full mt-4 text-xs">
        View All Alerts
      </Button>
    </CardContent>
  </Card>
);

export default AlertsPanel;