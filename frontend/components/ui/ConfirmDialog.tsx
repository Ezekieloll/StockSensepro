'use client';

import { CheckIcon, AlertIcon } from './Icons';

interface ConfirmDialogProps {
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    type?: 'danger' | 'warning' | 'info';
    onConfirm: () => void;
    onCancel: () => void;
}

export default function ConfirmDialog({
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    type = 'warning',
    onConfirm,
    onCancel
}: ConfirmDialogProps) {
    const colors = {
        danger: 'from-red-600 to-rose-600',
        warning: 'from-yellow-600 to-orange-600',
        info: 'from-indigo-600 to-purple-600'
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="glass border border-white/10 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl animate-scale-in">
                <div className="flex items-start gap-4 mb-4">
                    <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${colors[type]} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                        <AlertIcon size={24} className="text-white" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-foreground mb-2">{title}</h3>
                        <p className="text-sm text-muted">{message}</p>
                    </div>
                </div>
                <div className="flex gap-3 justify-end mt-6">
                    <button
                        onClick={onCancel}
                        className="px-6 py-2.5 rounded-lg bg-surface-elevated text-foreground border border-white/10 hover:bg-surface transition-all hover:-translate-y-0.5 font-semibold"
                    >
                        {cancelText}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-6 py-2.5 rounded-lg bg-gradient-to-r ${colors[type]} text-white hover:shadow-lg hover:shadow-${type === 'danger' ? 'red' : type === 'warning' ? 'orange' : 'indigo'}-500/50 transition-all hover:-translate-y-0.5 font-semibold`}
                    >
                        {confirmText}
                    </button>
                </div>
            </div>
        </div>
    );
}
