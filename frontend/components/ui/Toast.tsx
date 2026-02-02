'use client';

import { useEffect } from 'react';
import { CheckIcon, AlertIcon, InfoIcon, XIcon } from './Icons';

interface ToastProps {
    message: string;
    type?: 'success' | 'error' | 'warning' | 'info';
    onClose: () => void;
    duration?: number;
}

export default function Toast({ message, type = 'info', onClose, duration = 5000 }: ToastProps) {
    useEffect(() => {
        if (duration > 0) {
            const timer = setTimeout(onClose, duration);
            return () => clearTimeout(timer);
        }
    }, [duration, onClose]);

    const icons = {
        success: <CheckIcon size={20} />,
        error: <AlertIcon size={20} />,
        warning: <AlertIcon size={20} />,
        info: <InfoIcon size={20} />
    };

    const colors = {
        success: 'from-green-500/20 to-emerald-500/20 border-green-500/50 text-green-400',
        error: 'from-red-500/20 to-rose-500/20 border-red-500/50 text-red-400',
        warning: 'from-yellow-500/20 to-orange-500/20 border-yellow-500/50 text-yellow-400',
        info: 'from-blue-500/20 to-indigo-500/20 border-indigo-500/50 text-blue-400'
    };

    return (
        <div className={`fixed top-6 right-6 z-50 animate-slide-in-right`}>
            <div className={`glass border ${colors[type]} rounded-xl p-4 pr-12 shadow-2xl max-w-md backdrop-blur-xl`}>
                <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-0.5">
                        {icons[type]}
                    </div>
                    <div className="flex-1">
                        <p className="text-sm font-medium text-foreground whitespace-pre-line">{message}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="absolute top-3 right-3 text-muted hover:text-foreground transition-colors"
                    >
                        <XIcon size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
}
