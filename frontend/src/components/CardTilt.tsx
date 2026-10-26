import React, { useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface CardTiltProps {
  children: React.ReactNode;
  className?: string;
  maxTilt?: number;
  highlight?: boolean;
}

export const CardTilt: React.FC<CardTiltProps> = ({
  children,
  className = '',
  maxTilt = 7,
  highlight = true,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ rotateX: 0, rotateY: 0, spotlightX: 50, spotlightY: 50, isHovered: false });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;

    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const xPct = (mouseX / width) * 100;
    const yPct = (mouseY / height) * 100;

    const rotateX = -((mouseY - height / 2) / (height / 2)) * maxTilt;
    const rotateY = ((mouseX - width / 2) / (width / 2)) * maxTilt;

    setTilt({
      rotateX,
      rotateY,
      spotlightX: xPct,
      spotlightY: yPct,
      isHovered: true,
    });
  };

  const handleMouseLeave = () => {
    setTilt({
      rotateX: 0,
      rotateY: 0,
      spotlightX: 50,
      spotlightY: 50,
      isHovered: false,
    });
  };

  return (
    <div style={{ perspective: 1000 }} className="h-full">
      <motion.div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        animate={{
          rotateX: tilt.rotateX,
          rotateY: tilt.rotateY,
          scale: tilt.isHovered ? 1.01 : 1,
        }}
        transition={{ type: 'spring', stiffness: 260, damping: 20, mass: 0.1 }}
        style={{ transformStyle: 'preserve-3d' }}
        className={`relative overflow-hidden rounded-2xl ${className}`}
      >
        {/* Dynamic Specular Spotlight highlight */}
        {highlight && tilt.isHovered && (
          <div
            className="absolute inset-0 pointer-events-none transition-opacity duration-300 z-20"
            style={{
              background: `radial-gradient(circle 280px at ${tilt.spotlightX}% ${tilt.spotlightY}%, rgba(0, 242, 254, 0.12), transparent 70%)`,
            }}
          />
        )}
        <div style={{ transform: 'translateZ(20px)' }} className="relative z-10 h-full">
          {children}
        </div>
      </motion.div>
    </div>
  );
};
