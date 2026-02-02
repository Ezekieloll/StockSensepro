import React from 'react';

interface CardProps {
    children: React.ReactNode;
    className?: string;
    hover?: boolean;
    glass?: boolean;
}

export default function Card({ children, className = '', hover = true, glass = false }: CardProps) {
    const hoverClass = hover ? 'hover:border-primary/30 hover:shadow-lg hover:-translate-y-1' : '';
    const glassClass = glass ? 'glass-strong' : 'bg-surface';

    return (
        <div className={`${glassClass} rounded-xl p-6 border border-white/5 transition-smooth ${hoverClass} ${className}`}>
            {children}
        </div>
    );
}

export function CardHeader({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <div className={`mb-4 ${className}`}>{children}</div>;
}

export function CardTitle({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <h3 className={`text-xl font-bold text-foreground ${className}`}>{children}</h3>;
}

export function CardDescription({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <p className={`text-sm text-muted mt-1 ${className}`}>{children}</p>;
}

export function CardContent({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    return <div className={className}>{children}</div>;
}
