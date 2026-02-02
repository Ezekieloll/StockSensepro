import React from 'react';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    error?: string;
    icon?: React.ReactNode;
    options: { value: string; label: string }[];
}

export default function Select({
    label,
    error,
    icon,
    options,
    className = '',
    ...props
}: SelectProps) {
    return (
        <div className="w-full">
            {label && (
                <label className="block text-sm font-medium text-foreground mb-2">
                    {label}
                </label>
            )}
            <div className="relative">
                {icon && (
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none z-10">
                        {icon}
                    </div>
                )}
                <select
                    className={`input cursor-pointer ${icon ? 'pl-10' : ''} ${error ? 'border-error' : ''} ${className}`}
                    {...props}
                >
                    {options.map((option) => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
            </div>
            {error && (
                <p className="mt-1 text-sm text-error">{error}</p>
            )}
        </div>
    );
}
