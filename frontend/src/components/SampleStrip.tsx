interface SampleStripProps {
  onSelectSample: (modality: string, url: string) => void;
  disabled?: boolean;
}

const SAMPLES = [
  { modality: 'chest', label: 'Chest X-Ray', detail: 'Pulmonary Scan', icon: '🫁', file: '/samples/chest_sample.jpg' },
  { modality: 'brain', label: 'Brain MRI', detail: 'T1/T2 Axial Scan', icon: '🧠', file: '/samples/brain_sample.jpg' },
  { modality: 'eye', label: 'Eye Fundus', detail: 'Retinal Fundus', icon: '👁', file: '/samples/eye_sample.png' },
  { modality: 'skin', label: 'Skin Lesion', detail: 'Dermoscopy Scan', icon: '🔬', file: '/samples/skin_sample.jpg' },
];

export function SampleStrip({ onSelectSample, disabled }: SampleStripProps) {
  return (
    <div style={{
      marginBottom: '2rem',
      background: '#080808',
      border: '1px solid #262626',
      borderRadius: '12px',
      padding: '1.25rem 1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem',
    }}>
      <div>
        <span style={{
          fontSize: '0.62rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: '#00E5FF',
          display: 'block',
          marginBottom: '0.2rem',
        }}>
          Quick Test Benchmark
        </span>
        <span style={{ fontSize: '0.82rem', color: '#A3A3A3', fontWeight: 500 }}>
          Load verified clinical evaluation scans:
        </span>
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        {SAMPLES.map((s) => (
          <button
            key={s.modality}
            id={`sample-btn-${s.modality}`}
            onClick={() => onSelectSample(s.modality, s.file)}
            disabled={disabled}
            style={{
              padding: '0.6rem 1.1rem',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              background: '#0D0D0D',
              border: '1px solid #262626',
              borderRadius: '8px',
              color: '#F5F5F5',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
              transition: 'all 0.15s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
            onMouseEnter={(e) => {
              if (!disabled) {
                e.currentTarget.style.borderColor = '#00E5FF';
                e.currentTarget.style.background = '#121212';
                e.currentTarget.style.color = '#00E5FF';
              }
            }}
            onMouseLeave={(e) => {
              if (!disabled) {
                e.currentTarget.style.borderColor = '#262626';
                e.currentTarget.style.background = '#0D0D0D';
                e.currentTarget.style.color = '#F5F5F5';
              }
            }}
          >
            <span>{s.icon}</span>
            <span>{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
