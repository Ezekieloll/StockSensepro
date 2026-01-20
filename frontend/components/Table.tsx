import React from 'react';

export function Table({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return (
        <div className="w-full overflow-x-auto">
            <table className={`w-full text-sm text-left ${className}`}>
                {children}
            </table>
        </div>
    );
}

export function TableHeader({ children }: { children: React.ReactNode }) {
    return <thead className="text-xs uppercase bg-surface-elevated text-muted border-b border-white/5">{children}</thead>;
}

export function TableBody({ children }: { children: React.ReactNode }) {
    return <tbody className="divide-y divide-white/5">{children}</tbody>;
}

export function TableRow({ children, className = '', onClick }: { children: React.ReactNode; className?: string; onClick?: () => void }) {
    return <tr className={`hover:bg-white/[0.02] transition-colors ${onClick ? 'cursor-pointer' : ''} ${className}`} onClick={onClick}>{children}</tr>;
}

export function TableHead({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <th className={`px-6 py-4 font-medium text-muted ${className}`}>{children}</th>;
}

export function TableCell({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <td className={`px-6 py-4 ${className}`}>{children}</td>;
}
