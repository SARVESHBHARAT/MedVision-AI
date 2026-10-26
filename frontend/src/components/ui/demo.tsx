"use client";

import * as React from "react";
import HalftoneWordmarkFooter from "./halftone-wordmark-footer";
import Web3Bento from "./web3-bento";
import MedicalBento from "./medical-bento";

export default function Demo() {
  const [tab, setTab] = React.useState<"medical" | "web3" | "footer">("medical");

  return (
    <div className="dark min-h-screen w-full bg-background text-foreground p-4 sm:p-8 flex flex-col items-center">
      <div className="mb-8 flex items-center gap-2 rounded-full border border-border bg-card p-1">
        <button
          onClick={() => setTab("medical")}
          className={`rounded-full px-4 py-1.5 text-xs font-mono font-medium transition-colors ${tab === "medical" ? "bg-accent text-black font-bold" : "text-muted-foreground"}`}
        >
          Medical Bento (Combined)
        </button>
        <button
          onClick={() => setTab("web3")}
          className={`rounded-full px-4 py-1.5 text-xs font-mono font-medium transition-colors ${tab === "web3" ? "bg-accent text-black font-bold" : "text-muted-foreground"}`}
        >
          Raw 21st.dev Bento
        </button>
        <button
          onClick={() => setTab("footer")}
          className={`rounded-full px-4 py-1.5 text-xs font-mono font-medium transition-colors ${tab === "footer" ? "bg-accent text-black font-bold" : "text-muted-foreground"}`}
        >
          Halftone Footer
        </button>
      </div>

      <div className="w-full max-w-[1240px]">
        {tab === "medical" && <MedicalBento />}
        {tab === "web3" && <Web3Bento />}
        {tab === "footer" && <HalftoneWordmarkFooter wordmark="medvision" height="500px" />}
      </div>
    </div>
  );
}
