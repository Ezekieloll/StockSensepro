import React from 'react';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { PurchaseOrderItemCreate, ForecastAlert } from '../page';

interface PurchaseOrderModalProps {
  showPOModal: boolean;
  setShowPOModal: (show: boolean) => void;
  alerts: ForecastAlert[];
  poItems: PurchaseOrderItemCreate[];
  setPOItems: (items: PurchaseOrderItemCreate[]) => void;
  poNotes: string;
  setPONotes: (notes: string) => void;
  handleCreatePO: () => Promise<void>;
}

const PurchaseOrderModal: React.FC<PurchaseOrderModalProps> = ({
  showPOModal,
  setShowPOModal,
  alerts,
  poItems,
  setPOItems,
  poNotes,
  setPONotes,
  handleCreatePO,
}) => {
  if (!showPOModal) return null;
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setShowPOModal(false)}>
      <div className="bg-surface border border-white/10 rounded-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <h3 className="text-xl font-bold mb-4">Create Purchase Order</h3>
        <div className="mb-4">
          <label className="text-sm text-muted mb-2 block">Select SKUs from High-Risk Alerts</label>
          <div className="space-y-2 max-h-48 overflow-y-auto border border-white/10 rounded-lg p-3">
            {alerts.filter(a => a.severity === 'high').length === 0 ? (
              <p className="text-xs text-muted text-center py-4">No high-risk alerts. Add SKUs manually below.</p>
            ) : (
              alerts.filter(a => a.severity === 'high').slice(0, 10).map(alert => {
                const alreadyAdded = poItems.some(item => item.sku === alert.sku);
                return (
                  <div key={alert.id} className="flex items-center justify-between p-2 bg-white/5 rounded">
                    <div className="flex-1">
                      <span className="font-mono text-sm">{alert.sku}</span>
                      <span className="text-xs text-muted ml-2">Stock: {alert.current_stock}, Forecast: {alert.predicted_demand.toFixed(0)}</span>
                    </div>
                    <Button
                      variant={alreadyAdded ? "secondary" : "primary"}
                      size="sm"
                      onClick={() => {
                        if (!alreadyAdded) {
                          setPOItems([...poItems, {
                            sku: alert.sku,
                            quantity_requested: Math.ceil(alert.predicted_demand - alert.current_stock + 20),
                            unit_price: null
                          }]);
                        }
                      }}
                      disabled={alreadyAdded}
                    >
                      {alreadyAdded ? 'Added' : 'Add'}
                    </Button>
                  </div>
                );
              })
            )}
          </div>
        </div>
        <div className="mb-4">
          <label className="text-sm text-muted mb-2 block">Or Add SKU Manually</label>
          <div className="flex gap-2">
            <Input
              type="text"
              placeholder="Enter SKU (e.g., SKU_FRPR002)"
              className="flex-1"
              id="manual-sku-input"
            />
            <Input
              type="number"
              placeholder="Quantity"
              className="w-32"
              id="manual-qty-input"
            />
            <Button
              variant="primary"
              onClick={() => {
                const skuInput = document.getElementById('manual-sku-input') as HTMLInputElement;
                const qtyInput = document.getElementById('manual-qty-input') as HTMLInputElement;
                const sku = skuInput?.value.trim();
                const qty = parseFloat(qtyInput?.value);
                if (sku && qty > 0) {
                  const alreadyExists = poItems.some(item => item.sku === sku);
                  if (!alreadyExists) {
                    setPOItems([...poItems, {
                      sku: sku,
                      quantity_requested: qty,
                      unit_price: null
                    }]);
                    skuInput.value = '';
                    qtyInput.value = '';
                  } else {
                    alert('SKU already added!');
                  }
                } else {
                  alert('Please enter valid SKU and quantity');
                }
              }}
            >
              Add
            </Button>
          </div>
        </div>
        <div className="mb-4">
          <label className="text-sm text-muted mb-2 block">PO Items ({poItems.length})</label>
          {poItems.length === 0 ? (
            <p className="text-xs text-muted text-center py-8 border border-white/10 rounded-lg">No items added yet</p>
          ) : (
            <div className="space-y-2">
              {poItems.map((item, idx) => (
                <div key={idx} className="p-3 bg-white/5 border border-white/10 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-sm font-bold">{item.sku}</span>
                    <Button variant="ghost" size="sm" onClick={() => setPOItems(poItems.filter((_, i) => i !== idx))}>
                      Remove
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-muted block mb-1">Quantity</label>
                      <Input
                        type="number"
                        value={item.quantity_requested}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                          const newItems = [...poItems];
                          newItems[idx].quantity_requested = parseFloat(e.target.value) || 0;
                          setPOItems(newItems);
                        }}
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted block mb-1">Unit Price (optional)</label>
                      <Input
                        type="number"
                        step="0.01"
                        value={item.unit_price || ''}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                          const newItems = [...poItems];
                          newItems[idx].unit_price = parseFloat(e.target.value) || null;
                          setPOItems(newItems);
                        }}
                        placeholder="$0.00"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="mb-4">
          <label className="text-sm text-muted mb-1 block">Notes (optional)</label>
          <textarea
            value={poNotes}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setPONotes(e.target.value)}
            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-foreground resize-none"
            rows={3}
            placeholder="Add any special instructions or notes..."
          />
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={() => {
            setShowPOModal(false);
            setPOItems([]);
            setPONotes('');
          }} className="flex-1">
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleCreatePO}
            className="flex-1"
            disabled={poItems.length === 0}
          >
            Create PO ({poItems.length} items)
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PurchaseOrderModal;