import { createHashRouter } from "react-router";
import { Root } from "./components/Root";
import { Home } from "./components/pages/Home";
import { Onboarding } from "./components/pages/Onboarding";
import { RecordingFlow } from "./components/pages/RecordingFlow";
import { AudioLibrary } from "./components/pages/AudioLibrary";
import { AudioAnalysis } from "./components/pages/AudioAnalysis";

export const router = createHashRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: Home },
      { path: "onboarding", Component: Onboarding },
      { path: "record", Component: RecordingFlow },
      { path: "audio-library", Component: AudioLibrary },
      { path: "audio-analysis/:id", Component: AudioAnalysis },
    ],
  },
]);
