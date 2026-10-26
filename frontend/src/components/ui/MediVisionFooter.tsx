"use client";


export function MediVisionFooter() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <footer className="border-t border-[#262626] bg-[#000000] text-[#A3A3A3] pt-14 pb-10">
      <div className="mx-auto max-w-[1320px] px-6">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-4 lg:gap-12 pb-12 border-b border-[#181818]">
          {/* Brand & Mission Column */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-2.5">
              <span className="flex size-6 items-center justify-center rounded-md border border-[#383838] bg-[#0D0D0D] font-mono text-xs font-bold text-[#00E5FF]">
                ✦
              </span>
              <span className="font-mono text-base font-bold tracking-tight text-[#F5F5F5]">
                MediVision <span className="text-[#00E5FF]">AI</span>
              </span>
            </div>
            <p className="mt-3.5 text-xs text-[#737373] leading-relaxed max-w-[280px]">
              Multi-modal neural imaging and explainable AI platform. Transparent predictions with spatial attributions and automated clinical synthesis.
            </p>
            <div className="mt-4 flex items-center gap-2">
              <span className="size-2 rounded-full bg-[#34D399]" />
              <span className="font-mono text-[11px] text-[#A3A3A3]">CUDA Accelerated Pipeline</span>
            </div>
          </div>

          {/* Model Architectures Column */}
          <div>
            <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-[#F5F5F5] mb-3.5">
              Architectures
            </h4>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center justify-between text-[#A3A3A3]">
                <span>Brain MRI (v2 CNN)</span>
                <span className="font-mono text-[#34D399]">95.44%</span>
              </li>
              <li className="flex items-center justify-between text-[#A3A3A3]">
                <span>Chest X-Ray (DenseNet)</span>
                <span className="font-mono text-[#34D399]">92.95%</span>
              </li>
              <li className="flex items-center justify-between text-[#A3A3A3]">
                <span>Eye Fundus DR (CNN)</span>
                <span className="font-mono text-[#34D399]">90.5%</span>
              </li>
              <li className="flex items-center justify-between text-[#A3A3A3]">
                <span>Skin Lesion (EffNet)</span>
                <span className="font-mono text-[#34D399]">89.0%</span>
              </li>
            </ul>
          </div>

          {/* Explainability Suite Column */}
          <div>
            <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-[#F5F5F5] mb-3.5">
              Explainable AI
            </h4>
            <ul className="space-y-2 text-xs text-[#737373]">
              <li className="hover:text-[#F5F5F5] transition-colors">Grad-CAM Spatial Bottleneck Maps</li>
              <li className="hover:text-[#F5F5F5] transition-colors">LIME Superpixel Surrogate (0.8191 r)</li>
              <li className="hover:text-[#F5F5F5] transition-colors">SHAP Kernel Perturbation Robustness</li>
              <li className="hover:text-[#F5F5F5] transition-colors">Deep Q-Network Symptom Triage</li>
            </ul>
          </div>

          {/* Ethics & Back to Top */}
          <div className="flex flex-col justify-between">
            <div>
              <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-[#F5F5F5] mb-3.5">
                Clinical Use Notice
              </h4>
              <p className="text-[11px] text-[#737373] leading-relaxed">
                Research prototype. Predictions and heatmaps are decision-support aids for clinical investigators, not standalone medical diagnoses.
              </p>
            </div>
            <div className="pt-4">
              <button
                type="button"
                onClick={scrollToTop}
                className="inline-flex items-center gap-2 rounded-full border border-[#262626] bg-[#080808] px-4 py-2 text-xs font-mono font-medium text-[#A3A3A3] hover:border-[#383838] hover:text-[#F5F5F5] hover:bg-[#121212] transition-colors cursor-pointer"
              >
                Back to top ↑
              </button>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] text-[#525252]">
          <span>© 2026 MediVision AI Platform · All rights reserved</span>
          <span>PyTorch 2.6 · CUDA 12.4 · FastAPI</span>
        </div>
      </div>
    </footer>
  );
}

export default MediVisionFooter;
