import React, { useRef, useState } from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';

interface MagneticButtonProps extends HTMLMotionProps<'button'> {
  children: React.ReactNode;
  strength?: number;
  className?: string;
  glowColor?: string;
}

export const MagneticButton: React.FC<MagneticButtonProps> = ({
  children,
  strength = 25,
  className = '',
  glowColor = 'rgba(0, 242, 254, 0.4)',
  disabled = false,
  ...props
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (disabled || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const { clientX, clientY } = e;
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = (clientX - (rect.left + rect.width / 2)) * (strength / 100);
    const y = (clientY - (rect.top + rect.height / 2)) * (strength / 100);

    setPosition({ x, y });
  };

  const handleMouseLeave = () => {
    setPosition({ x: 0, y: 0 });
  };

  return (
    <motion.button
      ref={buttonRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      animate={{ x: position.x, y: position.y }}
      transition={{ type: 'spring', stiffness: 220, damping: 18, mass: 0.1 }}
      whileTap={{ scale: disabled ? 1 : 0.97 }}
      disabled={disabled}
      className={`relative inline-flex items-center justify-center font-medium transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...props}
    >
      {/* Subtle outer glow on hover */}
      <span
        className="absolute inset-0 rounded-xl opacity-0 transition-opacity duration-300 hover:opacity-100 pointer-events-none blur-lg"
        style={{ background: glowColor }}
      />
      <span className="relative z-10 flex items-center justify-center gap-2">
        {children}
      </span>
    </motion.button>
  );
};
