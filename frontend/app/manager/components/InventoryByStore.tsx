import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { DatabaseIcon } from '@/components/ui/Icons';
import React from 'react';

interface InventoryByStoreProps {
  inventoryByStore: any[];
}

const InventoryByStore: React.FC<InventoryByStoreProps> = ({ inventoryByStore }) => (
  <Card glass>
    <CardHeader>
      <div className="flex items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2 text-lg">
            <DatabaseIcon size={18} className="text-info" />
            Inventory by Store
          </CardTitle>
          <CardDescription>Stock health across locations</CardDescription>
        </div>
        <Button variant="ghost" size="sm">Manage</Button>
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-4">
        {inventoryByStore.map((store) => (
          <div key={store.store} className="p-4 bg-white/5 rounded-lg border border-white/5">
            <div className="flex items-center justify-between mb-3">
              <div className="font-medium">{store.store}</div>
              <div className="text-sm text-muted">${(store.value / 1000).toFixed(0)}K value</div>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-success rounded-full"></span>
                <span className="text-muted">{store.totalSKUs - store.lowStock - store.criticalStock} OK</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-warning rounded-full"></span>
                <span className="text-muted">{store.lowStock} Low</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-error rounded-full"></span>
                <span className="text-muted">{store.criticalStock} Critical</span>
              </div>
            </div>
            {/* Progress bar */}
            <div className="mt-3 h-2 bg-surface-elevated rounded-full overflow-hidden flex">
              <div
                className="bg-success h-full"
                style={{ width: `${((store.totalSKUs - store.lowStock - store.criticalStock) / store.totalSKUs) * 100}%` }}
              ></div>
              <div
                className="bg-warning h-full"
                style={{ width: `${(store.lowStock / store.totalSKUs) * 100}%` }}
              ></div>
              <div
                className="bg-error h-full"
                style={{ width: `${(store.criticalStock / store.totalSKUs) * 100}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
);

export default InventoryByStore;