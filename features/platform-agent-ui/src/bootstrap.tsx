import React from "react";
import { createRoot } from "react-dom/client";
import PlatformAgent from "./PlatformAgent";

const el = document.getElementById("root");
if (el) {
  createRoot(el).render(<PlatformAgent />);
}
