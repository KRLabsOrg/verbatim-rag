import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Utility function to merge class names with tailwind-merge
 * @param {...string} inputs - Class names to merge
 * @returns {string} Merged class names
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Common button variants using class-variance-authority patterns
 */
export const buttonVariants = {
  base: 'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background',
  variants: {
    variant: {
      default: 'bg-primary-600 text-white hover:bg-primary-700',
      secondary: 'bg-secondary-100 text-secondary-900 hover:bg-secondary-200',
      outline: 'border border-secondary-200 bg-white hover:bg-secondary-50',
      ghost: 'hover:bg-secondary-100',
      danger: 'bg-red-600 text-white hover:bg-red-700',
    },
    size: {
      default: 'h-10 py-2 px-4',
      sm: 'h-8 px-3 text-xs',
      lg: 'h-11 px-8',
      icon: 'h-10 w-10',
    },
  },
  defaultVariants: {
    variant: 'default',
    size: 'default',
  },
};

/**
 * Common input variants
 */
export const inputVariants = {
  base: 'flex h-10 w-full rounded-md border border-secondary-200 bg-white px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-secondary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
};

/**
 * Common card variants
 */
export const cardVariants = {
  base: 'rounded-lg border border-secondary-200 bg-white shadow-sm',
  variants: {
    elevated: 'shadow-md',
    flat: 'shadow-none',
  },
};

/**
 * Common badge variants
 */
export const badgeVariants = {
  base: 'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
  variants: {
    variant: {
      default: 'bg-primary-100 text-primary-800',
      secondary: 'bg-secondary-100 text-secondary-800',
      success: 'bg-green-100 text-green-800',
      warning: 'bg-yellow-100 text-yellow-800',
      danger: 'bg-red-100 text-red-800',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
};