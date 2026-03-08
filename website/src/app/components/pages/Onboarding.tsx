import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion, AnimatePresence } from "motion/react";
import { CheckCircle2, ChevronRight, ChevronLeft, Activity, ShieldCheck, HeartPulse } from "lucide-react";
import { Progress } from "../ui/progress";
import { Checkbox } from "../ui/checkbox";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

// ── Option lists ──────────────────────────────────────────────

const DIAGNOSTIC_GROUPS = [
  "Voice Disorder",
  "Neurological/Neurodegenerative Disorder",
  "Mood or Psychiatric Disorder",
  "Respiratory Disorder",
  "Multiple Disorders",
  "None (Control Group)",
] as const;

const VOICE_DISORDERS = [
  "Unilateral Vocal Fold Paralysis",
  "Laryngeal Dystonia",
  "Muscle Tension Dysphonia (MTD)",
  "Benign Lesions",
  "Laryngitis",
  "Laryngeal Cancer",
  "Glottic Insufficiency / Presbyphonia",
  "None of the above",
] as const;

const NEURO_DISORDERS = [
  "Alzheimer's / Dementia / MCI",
  "ALS",
  "Parkinson's Disease",
  "None of the above",
] as const;

const MOOD_DISORDERS = [
  "PTSD",
  "Depression / MDD",
  "Bipolar Disorder",
  "ADHD",
  "Anxiety Disorder",
  "None of the above",
] as const;

const RESPIRATORY_DISORDERS = [
  "COPD and Asthma",
  "Airway Stenosis",
  "Unexplained Chronic Cough",
  "None of the above",
] as const;

const GENDER_OPTIONS = ["Female", "Male", "Prefer not to answer"] as const;
const ORIENTATION_OPTIONS = ["Heterosexual", "Bisexual", "Homosexual", "Prefer not to answer", "Other"] as const;
const RACE_OPTIONS = [
  "American Indian or Alaska Native",
  "Asian",
  "Black or African American",
  "Native Hawaiian or Pacific Islander",
  "White",
  "Canadian Indigenous or Aboriginal",
  "Other",
  "Prefer not to answer",
] as const;
const ETHNICITY_OPTIONS = ["Not Hispanic or Latino", "Hispanic or Latino", "Prefer not to answer"] as const;
const LANGUAGE_OPTIONS = ["English", "Spanish", "Other"] as const;
const AGE_GROUPS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90 and above"] as const;

// ── Step identifiers ──────────────────────────────────────────

type StepId = "consent" | "diagnostic" | "voice" | "neuro" | "mood" | "respiratory" | "demographics";

// ── Helpers ───────────────────────────────────────────────────

function toggleInArray(arr: string[], value: string): string[] {
  return arr.includes(value) ? arr.filter(v => v !== value) : [...arr, value];
}

function toggleDisorderArray(arr: string[], value: string, noneLabel = "None of the above"): string[] {
  if (value === noneLabel) {
    return arr.includes(value) ? [] : [value];
  }
  const withoutNone = arr.filter(v => v !== noneLabel);
  const next = withoutNone.includes(value)
    ? withoutNone.filter(v => v !== value)
    : [...withoutNone, value];
  return next;
}

// ── Reusable sub-components ───────────────────────────────────

function CheckboxGroup({
  label,
  options,
  selected,
  onChange,
  toggleFn = toggleInArray,
}: {
  label: string;
  options: readonly string[];
  selected: string[];
  onChange: (next: string[]) => void;
  toggleFn?: (arr: string[], value: string) => string[];
}) {
  return (
    <div className="space-y-3">
      {label && <p className="text-sm font-medium text-purple-700">{label}</p>}
      <div className="space-y-2">
        {options.map(opt => (
          <label key={opt} className="flex items-center gap-3 p-3 bg-white/70 rounded-xl border border-purple-200/50 cursor-pointer hover:border-purple-400/60 hover:bg-white/90 transition-colors backdrop-blur-sm">
            <Checkbox
              checked={selected.includes(opt)}
              onCheckedChange={() => onChange(toggleFn(selected, opt))}
              className="border-purple-300 data-[state=checked]:bg-purple-600 data-[state=checked]:border-purple-600"
            />
            <span className="text-sm text-purple-900">{opt}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function RadioField({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-purple-700">{label}</p>
      <RadioGroup value={value} onValueChange={onChange} className="space-y-2">
        {options.map(opt => (
          <label key={opt} className="flex items-center gap-3 p-3 bg-white/70 rounded-xl border border-purple-200/50 cursor-pointer hover:border-purple-400/60 hover:bg-white/90 transition-colors backdrop-blur-sm">
            <RadioGroupItem value={opt} className="border-purple-300 text-purple-600" />
            <Label className="text-sm text-purple-900 cursor-pointer">{opt}</Label>
          </label>
        ))}
      </RadioGroup>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────

export function Onboarding() {
  const { userData, setUserData, surveyAnswers, setSurveyAnswer } = useAppContext();
  const navigate = useNavigate();
  const [agreed, setAgreed] = useState(false);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);

  const steps = useMemo<StepId[]>(() => {
    const s: StepId[] = ["consent", "diagnostic"];
    const dg = surveyAnswers.diagnosticGroups;
    if (dg.includes("Voice Disorder")) s.push("voice");
    if (dg.includes("Neurological/Neurodegenerative Disorder")) s.push("neuro");
    if (dg.includes("Mood or Psychiatric Disorder")) s.push("mood");
    if (dg.includes("Respiratory Disorder")) s.push("respiratory");
    s.push("demographics");
    return s;
  }, [surveyAnswers.diagnosticGroups]);

  const currentStep = steps[currentStepIdx] ?? "consent";
  const totalSteps = steps.length;
  const progressPercent = ((currentStepIdx + 1) / totalSteps) * 100;

  if (currentStepIdx >= steps.length) {
    setCurrentStepIdx(steps.length - 1);
  }

  const canAdvance = (): boolean => {
    switch (currentStep) {
      case "consent": return agreed;
      case "diagnostic": return surveyAnswers.diagnosticGroups.length > 0;
      case "voice": return surveyAnswers.voiceDisorders.length > 0;
      case "neuro": return surveyAnswers.neurologicalDisorders.length > 0;
      case "mood": return surveyAnswers.moodDisorders.length > 0;
      case "respiratory": return surveyAnswers.respiratoryDisorders.length > 0;
      case "demographics":
        return !!(surveyAnswers.genderIdentity && surveyAnswers.sexualOrientation && surveyAnswers.race.length > 0 && surveyAnswers.ethnicity && surveyAnswers.primaryLanguage && surveyAnswers.ageGroup);
      default: return false;
    }
  };

  const handleNext = () => {
    if (!canAdvance()) return;
    if (currentStepIdx < totalSteps - 1) {
      setCurrentStepIdx(currentStepIdx + 1);
    } else {
      setUserData({ ...userData, optedIn: true, onboardingComplete: true, demographics: surveyAnswers, hasRecordedToday: false });
      navigate("/");
    }
  };

  const handleBack = () => {
    if (currentStepIdx > 0) setCurrentStepIdx(currentStepIdx - 1);
  };

  const handleDiagnosticChange = (next: string[]) => {
    setSurveyAnswer("diagnosticGroups", next);
    if (!next.includes("Voice Disorder")) setSurveyAnswer("voiceDisorders", []);
    if (!next.includes("Neurological/Neurodegenerative Disorder")) setSurveyAnswer("neurologicalDisorders", []);
    if (!next.includes("Mood or Psychiatric Disorder")) setSurveyAnswer("moodDisorders", []);
    if (!next.includes("Respiratory Disorder")) setSurveyAnswer("respiratoryDisorders", []);
  };

  // ── Step renderers ────────────────────────────────────────

  const renderConsent = () => (
    <div className="flex-1 space-y-8 mt-12">
      <div className="flex justify-center">
        <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center border border-purple-200/60">
          <Activity className="w-10 h-10 text-purple-600" />
        </div>
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-center text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Bridge2AI Study</h1>
      <p className="text-purple-500 text-center leading-relaxed">
        We're mapping voice biomarkers to predict health changes. Your daily recordings help build the future of diagnostic medicine with unprecedented accuracy.
      </p>
      <div className="space-y-4 bg-white/60 p-4 rounded-2xl border border-purple-200/50 backdrop-blur-sm">
        <div className="flex gap-3 items-start">
          <ShieldCheck className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" />
          <p className="text-sm text-purple-700">Your voice data is encrypted and completely anonymized.</p>
        </div>
        <div className="flex gap-3 items-start">
          <HeartPulse className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
          <p className="text-sm text-purple-700">Discover early signs of vocal strain or fatigue before it happens.</p>
        </div>
      </div>
      <label className="flex gap-3 items-center p-4 bg-white/70 rounded-2xl border border-purple-200/50 cursor-pointer hover:border-purple-400/60 transition-colors backdrop-blur-sm">
        <div className={`w-6 h-6 rounded-md flex items-center justify-center border transition-colors ${agreed ? "bg-purple-600 border-purple-600 text-white" : "border-purple-300 text-transparent"}`}>
          <CheckCircle2 className="w-4 h-4" />
        </div>
        <span className="text-sm text-purple-800">I agree to contribute to the Bridge2AI voice study.</span>
        <input type="checkbox" className="hidden" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
      </label>
      <button
        type="button"
        onClick={() => {
          setUserData(prev => ({ ...prev, optedIn: false, onboardingComplete: true }));
          navigate("/");
        }}
        className="w-full py-3 text-sm text-purple-400 hover:text-rose-500 transition-colors underline underline-offset-2"
      >
        I do not wish to participate — Opt out of Bridge2AI study
      </button>
    </div>
  );

  const renderDiagnostic = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Diagnostic Group</h1>
        <p className="text-purple-500 mt-2">Do you have any diagnosed conditions? Select all that apply.</p>
      </div>
      <CheckboxGroup label="" options={DIAGNOSTIC_GROUPS} selected={surveyAnswers.diagnosticGroups} onChange={handleDiagnosticChange} />
    </div>
  );

  const renderVoice = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Voice Disorder</h1>
        <p className="text-purple-500 mt-2">Which voice condition(s) apply to you?</p>
      </div>
      <CheckboxGroup label="" options={VOICE_DISORDERS} selected={surveyAnswers.voiceDisorders} onChange={(v) => setSurveyAnswer("voiceDisorders", v)} toggleFn={(arr, val) => toggleDisorderArray(arr, val)} />
    </div>
  );

  const renderNeuro = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Neurological Disorder</h1>
        <p className="text-purple-500 mt-2">Which neurological condition(s) apply to you?</p>
      </div>
      <CheckboxGroup label="" options={NEURO_DISORDERS} selected={surveyAnswers.neurologicalDisorders} onChange={(v) => setSurveyAnswer("neurologicalDisorders", v)} toggleFn={(arr, val) => toggleDisorderArray(arr, val)} />
    </div>
  );

  const renderMood = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Mood & Psychiatric</h1>
        <p className="text-purple-500 mt-2">Which mood or psychiatric condition(s) apply to you?</p>
      </div>
      <CheckboxGroup label="" options={MOOD_DISORDERS} selected={surveyAnswers.moodDisorders} onChange={(v) => setSurveyAnswer("moodDisorders", v)} toggleFn={(arr, val) => toggleDisorderArray(arr, val)} />
    </div>
  );

  const renderRespiratory = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Respiratory Disorder</h1>
        <p className="text-purple-500 mt-2">Which respiratory condition(s) apply to you?</p>
      </div>
      <CheckboxGroup label="" options={RESPIRATORY_DISORDERS} selected={surveyAnswers.respiratoryDisorders} onChange={(v) => setSurveyAnswer("respiratoryDisorders", v)} toggleFn={(arr, val) => toggleDisorderArray(arr, val)} />
    </div>
  );

  const renderDemographics = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Demographics</h1>
        <p className="text-purple-500 mt-2">Help us contextualize your biomarkers with a few baseline details.</p>
      </div>
      <div className="space-y-6">
        <RadioField label="Gender Identity" options={GENDER_OPTIONS} value={surveyAnswers.genderIdentity} onChange={(v) => setSurveyAnswer("genderIdentity", v)} />
        <RadioField label="Sexual Orientation" options={ORIENTATION_OPTIONS} value={surveyAnswers.sexualOrientation} onChange={(v) => setSurveyAnswer("sexualOrientation", v)} />
        <CheckboxGroup label="Race (select all that apply)" options={RACE_OPTIONS} selected={surveyAnswers.race} onChange={(v) => setSurveyAnswer("race", v)} />
        <RadioField label="Ethnicity" options={ETHNICITY_OPTIONS} value={surveyAnswers.ethnicity} onChange={(v) => setSurveyAnswer("ethnicity", v)} />
        <RadioField label="Primary Language" options={LANGUAGE_OPTIONS} value={surveyAnswers.primaryLanguage} onChange={(v) => setSurveyAnswer("primaryLanguage", v)} />
        <div className="space-y-3">
          <p className="text-sm font-medium text-purple-700">Age Group</p>
          <Select value={surveyAnswers.ageGroup} onValueChange={(v) => setSurveyAnswer("ageGroup", v)}>
            <SelectTrigger className="w-full bg-white/70 border-purple-200/50 text-purple-900 backdrop-blur-sm">
              <SelectValue placeholder="Select age group..." />
            </SelectTrigger>
            <SelectContent>
              {AGE_GROUPS.map(g => (
                <SelectItem key={g} value={g}>{g}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );

  const stepContent: Record<StepId, () => React.ReactNode> = {
    consent: renderConsent,
    diagnostic: renderDiagnostic,
    voice: renderVoice,
    neuro: renderNeuro,
    mood: renderMood,
    respiratory: renderRespiratory,
    demographics: renderDemographics,
  };

  const isLastStep = currentStepIdx === totalSteps - 1;

  return (
    <div className="flex flex-col h-full text-purple-950 p-6 overflow-y-auto relative z-10" style={{ fontFamily: "var(--font-body)" }}>
      {/* Progress bar — hidden on consent step */}
      {currentStep !== "consent" && (
        <div className="mb-4 space-y-2">
          <div className="flex justify-between text-xs text-purple-400">
            <span>Step {currentStepIdx + 1} of {totalSteps}</span>
            <span>{Math.round(progressPercent)}%</span>
          </div>
          <Progress value={progressPercent} className="bg-purple-200/40 h-2" />
        </div>
      )}

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="flex flex-col flex-1"
        >
          {stepContent[currentStep]()}

          <div className="flex gap-3 mt-8">
            {currentStepIdx > 0 && (
              <button
                onClick={handleBack}
                className="flex-1 py-4 bg-white/60 text-purple-700 rounded-2xl font-semibold hover:bg-white/80 transition-colors flex justify-center items-center gap-2 border border-purple-200/50 backdrop-blur-sm"
              >
                <ChevronLeft className="w-5 h-5" /> Back
              </button>
            )}
            <button
              onClick={handleNext}
              disabled={!canAdvance()}
              className="flex-1 py-4 bg-purple-600 text-white rounded-2xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-purple-700 transition-colors flex justify-center items-center gap-2 shadow-lg shadow-purple-500/20"
            >
              {isLastStep ? (
                <>Complete Calibration <CheckCircle2 className="w-5 h-5" /></>
              ) : (
                <>Continue <ChevronRight className="w-5 h-5" /></>
              )}
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
