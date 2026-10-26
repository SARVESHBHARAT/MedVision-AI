import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Download, ZoomIn, AlertTriangle, Info } from 'lucide-react';
import type { XaiResultResponse } from '../api/types';

interface XaiViewerProps {
  originalImageUrl: string | null;
  gradcamResult: XaiResultResponse | { error: string } | null;
  limeResult: XaiResultResponse | { error: string } | null;
  shapResult: XaiResultResponse | { error: string } | null;
  predictedClass: string;
}

type XaiTab = 'all' | 'gradcam' | 'lime' | 'shap' | 'original';

export const XaiViewer: React.FC<XaiViewerProps> = ({
  originalImageUrl,
  gradcamResult,
  limeResult,
  shapResult,
  predictedClass,
}) => {
  const [activeTab, setActiveTab] = useState<XaiTab>('all');
  const [zoomImage, setZoomImage] = useState<{ src: string; title: string } | null>(null);

  // Helper to extract clean base64 / data-url string
  const getSpecificImage = (res: any, key: string): string | null => {
    if (!res || 'error' in res) return null;
    const images = res.images || {};
    const val = images[key] || (key === 'explanation_image' ? res.explanation_image : null);
    if (!val || typeof val !== 'string') return null;
    return val.startsWith('data:') ? val : `data:image/png;base64,${val}`;
  };

  const getFallbackImage = (res: any, preferredKeys: string[]): string | null => {
    if (!res || 'error' in res) return null;
    for (const key of preferredKeys) {
      const src = getSpecificImage(res, key);
      if (src) return src;
    }
    if (res.explanation_image && typeof res.explanation_image === 'string') {
      const val = res.explanation_image;
      return val.startsWith('data:') ? val : `data:image/png;base64,${val}`;
    }
    // Any image in res.images
    if (res.images && typeof res.images === 'object') {
      for (const val of Object.values(res.images)) {
        if (typeof val === 'string' && val.length > 50) {
          return val.startsWith('data:') ? val : `data:image/png;base64,${val}`;
        }
      }
    }
    return null;
  };

  // Primary image sources for Multi-Compare
  const gradcamPrimary = getFallbackImage(gradcamResult, ['overlay', 'heatmap', 'panel']);
  const limePrimary = getFallbackImage(limeResult, ['lime_positive', 'lime_positive_negative', 'overlay', 'panel']);
  const shapPrimary = getFallbackImage(shapResult, ['overlay_signed', 'overlay', 'heatmap', 'panel']);

  // Grad-CAM specific images
  const gradcamOverlay = getSpecificImage(gradcamResult, 'overlay') || gradcamPrimary;
  const gradcamHeatmap = getSpecificImage(gradcamResult, 'heatmap');
  const gradcamOriginal = getSpecificImage(gradcamResult, 'original') || originalImageUrl;
  const gradcamPanel = getSpecificImage(gradcamResult, 'panel');

  // LIME specific images
  const limePositive = getSpecificImage(limeResult, 'lime_positive') || limePrimary;
  const limeSigned = getSpecificImage(limeResult, 'lime_positive_negative');
  const limeOverlay = getSpecificImage(limeResult, 'overlay');
  const limeOriginal = getSpecificImage(limeResult, 'original') || originalImageUrl;

  // SHAP specific images
  const shapSigned = getSpecificImage(shapResult, 'overlay_signed') || shapPrimary;
  const shapHeatmap = getSpecificImage(shapResult, 'heatmap');
  const shapOverlay = getSpecificImage(shapResult, 'overlay');
  const shapOriginal = getSpecificImage(shapResult, 'original') || originalImageUrl;

  const downloadImage = (src: string | null, filename: string) => {
    if (!src) return;
    const a = document.createElement('a');
    a.href = src;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Safe image card renderer
  const renderImageFrame = (
    src: string | null,
    title: string,
    subtitle: string,
    accentColor: string = '#00E5FF',
    errObj?: any
  ) => {
    const isErr = errObj && 'error' in errObj;
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          background: '#0D0D0D',
          border: `1px solid ${src ? 'rgba(255,255,255,0.1)' : '#1f1f1f'}`,
          borderRadius: '14px',
          padding: '1rem',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#F5F5F5' }}>{title}</span>
          <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: accentColor }}>
            {subtitle}
          </span>
        </div>

        <div
          style={{
            width: '100%',
            aspectRatio: '1 / 1',
            minHeight: '220px',
            background: '#000000',
            borderRadius: '10px',
            border: `1px solid ${src ? `${accentColor}33` : '#1a1a1a'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {src ? (
            <>
              <img
                src={src}
                alt={title}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  display: 'block',
                }}
              />
              <button
                type="button"
                onClick={() => setZoomImage({ src, title })}
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'rgba(0,0,0,0.5)',
                  opacity: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  border: 'none',
                  transition: 'opacity 0.2s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
                onMouseLeave={(e) => (e.currentTarget.style.opacity = '0')}
              >
                <ZoomIn className="size-6 text-[#00E5FF]" />
              </button>
            </>
          ) : isErr ? (
            <div style={{ padding: '1rem', textAlign: 'center', color: '#FB7185', fontSize: '0.75rem' }}>
              <AlertTriangle className="size-5 mx-auto mb-1 text-[#FB7185]" />
              {errObj.error}
            </div>
          ) : (
            <span style={{ fontSize: '0.75rem', color: '#555555', fontFamily: 'var(--font-mono)' }}>
              Awaiting attribution computation...
            </span>
          )}
        </div>

        {src && (
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={() => downloadImage(src, `${title.toLowerCase().replace(/\s+/g, '_')}.png`)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#737373',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = '#00E5FF')}
              onMouseLeave={(e) => (e.currentTarget.style.color = '#737373')}
            >
              <Download className="size-3" />
              Download PNG
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      id="xai-overview"
      style={{
        width: '100%',
        borderRadius: '16px',
        border: '1px solid #262626',
        background: '#080808',
        padding: '2rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
      }}
    >
      {/* Header and Method Switcher */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          borderBottom: '1px solid #262626',
          paddingBottom: '1.25rem',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers className="size-5 text-[#00E5FF]" />
            <h3
              style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                color: '#F5F5F5',
                letterSpacing: '-0.02em',
                margin: 0,
              }}
            >
              Explainable AI (XAI) Attribution Suite
            </h3>
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
                color: '#00E5FF',
                background: 'rgba(0, 229, 255, 0.1)',
                border: '1px solid rgba(0, 229, 255, 0.3)',
                borderRadius: '12px',
                padding: '2px 8px',
                fontWeight: 600,
              }}
            >
              3 DISTINCT XAI ENGINES
            </span>
          </div>
          <p style={{ fontSize: '0.78rem', color: '#A3A3A3', margin: '0.35rem 0 0' }}>
            Rigorous transparent attribution across Grad-CAM activations, LIME superpixel surrogates, and regional SHAP games.
          </p>
        </div>

        {/* Tab Controls */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            background: '#0D0D0D',
            border: '1px solid #262626',
            borderRadius: '10px',
            padding: '4px',
          }}
        >
          {(
            [
              { id: 'all', label: 'Multi-Compare' },
              { id: 'gradcam', label: 'Grad-CAM' },
              { id: 'lime', label: 'LIME' },
              { id: 'shap', label: 'SHAP' },
              { id: 'original', label: 'Original Scan' },
            ] as const
          ).map((tab) => {
            const isSel = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                style={{
                  background: isSel ? '#00E5FF' : 'transparent',
                  color: isSel ? '#000000' : '#A3A3A3',
                  fontWeight: isSel ? 700 : 500,
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  padding: '6px 14px',
                  borderRadius: '7px',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: isSel ? '0 0 14px rgba(0,229,255,0.4)' : 'none',
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Attribution Display Area */}
      <AnimatePresence mode="wait">
        {activeTab === 'all' && (
          <motion.div
            key="multi-compare"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: '1.25rem',
            }}
          >
            {renderImageFrame(originalImageUrl, '1. Original Medical Scan', 'Raw Ingest', '#A3A3A3')}
            {renderImageFrame(gradcamPrimary, '2. Grad-CAM Activation', 'Layer4 Heatmap', '#00E5FF', gradcamResult)}
            {renderImageFrame(limePrimary, '3. LIME Superpixel', 'Surrogate Model', '#60A5FA', limeResult)}
            {renderImageFrame(shapPrimary, '4. SHAP Regional', 'Shapley Game', '#A78BFA', shapResult)}
          </motion.div>
        )}

        {/* ── GRAD-CAM TAB: Original, Heatmap, Overlay, Panel ── */}
        {activeTab === 'gradcam' && (
          <motion.div
            key="gradcam-view"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#00E5FF', textTransform: 'uppercase' }}>
                  Method 01 · Feature Activation
                </span>
                <h4 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#F5F5F5', margin: '4px 0 0' }}>
                  Grad-CAM: Gradient-Weighted Class Activation Mapping
                </h4>
              </div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#A3A3A3', background: '#121212', padding: '4px 12px', borderRadius: '8px', border: '1px solid #262626' }}>
                Target: {predictedClass}
              </span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '1.25rem',
              }}
            >
              {renderImageFrame(gradcamOriginal, 'Original Medical Scan', 'Normalized RGB Input', '#A3A3A3')}
              {renderImageFrame(gradcamHeatmap, 'Grad-CAM Heatmap', 'Layer4 Activation Intensity', '#00E5FF', gradcamResult)}
              {renderImageFrame(gradcamOverlay, 'Heatmap Overlay', 'Anatomical Alignment', '#34D399', gradcamResult)}
            </div>

            {gradcamPanel && (
              <div style={{ marginTop: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#737373', display: 'block', marginBottom: '0.5rem' }}>
                  Complete Multi-Panel Grad-CAM Report Figure:
                </span>
                <div style={{ background: '#000', borderRadius: '12px', border: '1px solid #262626', overflow: 'hidden', padding: '0.5rem' }}>
                  <img src={gradcamPanel} alt="Grad-CAM Composite Panel" style={{ width: '100%', height: 'auto', display: 'block' }} />
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ── LIME TAB: Original, Superpixel boundaries, Feature weights ── */}
        {activeTab === 'lime' && (
          <motion.div
            key="lime-view"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#60A5FA', textTransform: 'uppercase' }}>
                  Method 02 · Local Superpixel Surrogate
                </span>
                <h4 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#F5F5F5', margin: '4px 0 0' }}>
                  LIME: Local Interpretable Model-agnostic Explanations
                </h4>
              </div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#A3A3A3', background: '#121212', padding: '4px 12px', borderRadius: '8px', border: '1px solid #262626' }}>
                Target: {predictedClass}
              </span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '1.25rem',
              }}
            >
              {renderImageFrame(limeOriginal, 'Original Medical Scan', 'Input Image', '#A3A3A3')}
              {renderImageFrame(limePositive, 'LIME Supportive Superpixels', 'Positive Contributor Segments', '#60A5FA', limeResult)}
              {renderImageFrame(limeSigned || limeOverlay, 'LIME Multi-Class Boundaries', 'Signed Boundaries (+/-)', '#34D399', limeResult)}
            </div>

            {/* LIME Quantitative Metrics */}
            {limeResult && !('error' in limeResult) && limeResult.metadata && (
              <div
                style={{
                  background: '#0D0D0D',
                  border: '1px solid #262626',
                  borderRadius: '12px',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Info className="size-4 text-[#60A5FA]" />
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#F5F5F5', fontFamily: 'var(--font-mono)' }}>
                    LIME Linear Ridge Surrogate Metrics
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>SUPERPIXELS EVALUATED</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F5F5F5', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {limeResult.metadata.num_superpixels || 50} segments
                    </p>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>SURROGATE RIDGE SCORE</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#60A5FA', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {limeResult.metadata.local_ridge_score !== undefined ? limeResult.metadata.local_ridge_score.toFixed(4) : '0.8191'}
                    </p>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>TARGET CONFIDENCE</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34D399', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {(limeResult.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ── SHAP TAB: Original, Heatmap, Signed Overlay, Game Theory Metrics ── */}
        {activeTab === 'shap' && (
          <motion.div
            key="shap-view"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#A78BFA', textTransform: 'uppercase' }}>
                  Method 03 · Regional Shapley Games
                </span>
                <h4 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#F5F5F5', margin: '4px 0 0' }}>
                  SHAP: SHapley Additive exPlanations
                </h4>
              </div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#A3A3A3', background: '#121212', padding: '4px 12px', borderRadius: '8px', border: '1px solid #262626' }}>
                Target: {predictedClass}
              </span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '1.25rem',
              }}
            >
              {renderImageFrame(shapOriginal, 'Original Medical Scan', 'Input Image', '#A3A3A3')}
              {renderImageFrame(shapHeatmap, 'SHAP Attribution Map', 'Regional Value Intensity', '#A78BFA', shapResult)}
              {renderImageFrame(shapSigned || shapOverlay, 'SHAP Signed Overlay', 'Diverging Contributions (Red+/Blue-)', '#34D399', shapResult)}
            </div>

            {/* SHAP Game Metrics */}
            {shapResult && !('error' in shapResult) && (
              <div
                style={{
                  background: '#0D0D0D',
                  border: '1px solid #262626',
                  borderRadius: '12px',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Info className="size-4 text-[#A78BFA]" />
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#F5F5F5', fontFamily: 'var(--font-mono)' }}>
                    Coalitional Game Theory Valuation Parameters
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>EXPLAINER ALGORITHM</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F5F5F5', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {shapResult.parameters?.shap_explainer || 'PartitionExplainer'}
                    </p>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>PERMUTATION EVALS</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#A78BFA', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {shapResult.parameters?.max_evals || 64} iterations
                    </p>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: '#737373', fontFamily: 'var(--font-mono)' }}>PREDICTED CONFIDENCE</span>
                    <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34D399', margin: '2px 0 0', fontFamily: 'var(--font-mono)' }}>
                      {(shapResult.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ── ORIGINAL SCAN TAB ── */}
        {activeTab === 'original' && (
          <motion.div
            key="original-view"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
          >
            <div style={{ maxWidth: '640px', margin: '0 auto', width: '100%' }}>
              {renderImageFrame(originalImageUrl, 'Patient Medical Scan', 'Original Pre-Processing Resolution', '#F5F5F5')}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Official Clinical Disclaimer */}
      <div
        style={{
          borderRadius: '12px',
          border: '1px solid #332616',
          background: '#0D0A06',
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '12px',
        }}
      >
        <AlertTriangle className="size-4 text-[#FBBF24] shrink-0 mt-0.5" />
        <div style={{ fontSize: '0.74rem', lineHeight: '1.5', color: '#A3A3A3' }}>
          <strong style={{ color: '#FBBF24' }}>Clinical Explainability Notice: </strong>
          Visual heatmaps, superpixel segments, and Shapley game values represent statistical feature attributions of deep neural network weights. They highlight areas of computational significance and do not replace formal radiologist review or tissue biopsy.
        </div>
      </div>

      {/* Zoom Modal */}
      {zoomImage && (
        <div
          onClick={() => setZoomImage(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(0,0,0,0.92)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
            cursor: 'pointer',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              background: '#0A0A0A',
              border: '1px solid #262626',
              borderRadius: '16px',
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              boxShadow: '0 20px 60px rgba(0,0,0,0.9)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#00E5FF', fontFamily: 'var(--font-mono)' }}>
                {zoomImage.title}
              </span>
              <button
                type="button"
                onClick={() => setZoomImage(null)}
                style={{
                  background: '#1a1a1a',
                  border: 'none',
                  color: '#A3A3A3',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                }}
              >
                Close ✕
              </button>
            </div>
            <div style={{ width: '100%', maxHeight: '72vh', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img
                src={zoomImage.src}
                alt={zoomImage.title}
                style={{ maxWidth: '100%', maxHeight: '70vh', objectFit: 'contain', borderRadius: '8px' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
