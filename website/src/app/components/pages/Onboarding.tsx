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

/** For disorder groups: "None of the above" is mutually exclusive with other options */
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
      {label && <p className="text-sm font-medium text-slate-600">{label}</p>}
      <div className="space-y-2">
        {options.map(opt => (
          <label key={opt} className="flex items-center gap-3 p-3 bg-white rounded-xl border border-violet-100 cursor-pointer hover:border-violet-300 transition-colors shadow-sm">
            <Checkbox
              checked={selected.includes(opt)}
              onCheckedChange={() => onChange(toggleFn(selected, opt))}
              className="border-violet-300 data-[state=checked]:bg-violet-600 data-[state=checked]:border-violet-600"
            />
            <span className="text-sm text-slate-700">{opt}</span>
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
      <p className="text-sm font-medium text-slate-600">{label}</p>
      <RadioGroup value={value} onValueChange={onChange} className="space-y-2">
        {options.map(opt => (
          <label key={opt} className="flex items-center gap-3 p-3 bg-white rounded-xl border border-violet-100 cursor-pointer hover:border-violet-300 transition-colors shadow-sm">
            <RadioGroupItem value={opt} className="border-violet-300 text-violet-600" />
            <Label className="text-sm text-slate-700 cursor-pointer">{opt}</Label>
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

  // Build the dynamic step list based on diagnostic selections
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

  // Clamp index if steps shrink (e.g. user goes back and deselects a group)
  if (currentStepIdx >= steps.length) {
    setCurrentStepIdx(steps.length - 1);
  }

  const canAdvance = (): boolean => {
    switch (currentStep) {
      case "consent":
        return agreed;
      case "diagnostic":
        return surveyAnswers.diagnosticGroups.length > 0;
      case "voice":
        return surveyAnswers.voiceDisorders.length > 0;
      case "neuro":
        return surveyAnswers.neurologicalDisorders.length > 0;
      case "mood":
        return surveyAnswers.moodDisorders.length > 0;
      case "respiratory":
        return surveyAnswers.respiratoryDisorders.length > 0;
      case "demographics":
        return !!(
          surveyAnswers.genderIdentity &&
          surveyAnswers.sexualOrientation &&
          surveyAnswers.race.length > 0 &&
          surveyAnswers.ethnicity &&
          surveyAnswers.primaryLanguage &&
          surveyAnswers.ageGroup
        );
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (!canAdvance()) return;
    if (currentStepIdx < totalSteps - 1) {
      setCurrentStepIdx(currentStepIdx + 1);
    } else {
      // Final step — complete onboarding (opt in)
      setUserData({ ...userData, optedIn: true, onboardingComplete: true, demographics: surveyAnswers, hasRecordedToday: false });
      navigate("/");
    }
  };

  const handleBack = () => {
    if (currentStepIdx > 0) setCurrentStepIdx(currentStepIdx - 1);
  };

  // When diagnostic groups change, clear sub-answers for deselected groups
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
        <div className="w-20 h-20 bg-violet-100 rounded-full flex items-center justify-center">
          <Activity className="w-10 h-10 text-violet-600" />
        </div>
      </div>
      <h1 className="text-5xl font-extrabold tracking-tight text-center">AriaPitch</h1>
      <p className="text-slate-500 text-center leading-relaxed">
        We're mapping voice biomarkers to predict health changes. Your daily recordings help build the future of diagnostic medicine with unprecedented accuracy.
      </p>
      <div className="space-y-4 bg-violet-50/60 p-4 rounded-2xl border border-violet-100">
        <div className="flex gap-3 items-start">
          <ShieldCheck className="w-5 h-5 text-violet-600 shrink-0 mt-0.5" />
          <p className="text-sm text-slate-600">Your data is completely anonymized.</p>
        </div>
        <div className="flex gap-3 items-start">
          <HeartPulse className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
          <p className="text-sm text-slate-600">Discover early signs of vocal strain or fatigue before it happens.</p>
        </div>
      </div>
      <p className="text-sm text-slate-400 text-center">
        Powered by{" "}
        <a href="https://b2ai-voice.org" target="_blank" rel="noopener noreferrer" className="text-violet-500 hover:text-violet-700 font-semibold underline underline-offset-2 transition-colors">
          Bridge2AI Voice
        </a>
      </p>
      <button
        type="button"
        onClick={() => setAgreed(true)}
        className={`w-full py-4 text-base font-medium rounded-2xl transition-colors ${agreed ? "bg-violet-600 text-white border border-violet-600" : "text-violet-600 border border-violet-200 hover:border-violet-400 hover:bg-violet-50"}`}
      >
        I wish to participate
      </button>
      <button
        type="button"
        onClick={() => {
          setUserData(prev => ({ ...prev, optedIn: false, onboardingComplete: true }));
          navigate("/");
        }}
        className="w-full py-4 text-base font-medium text-slate-500 hover:text-rose-500 transition-colors border border-slate-200 hover:border-rose-200 rounded-2xl"
      >
        I do not wish to participate — continue to recording
      </button>
    </div>
  );

  const renderDiagnostic = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Diagnostic Group</h1>
        <p className="text-slate-500 mt-2">Do you have any diagnosed conditions? Select all that apply.</p>
      </div>
      <CheckboxGroup
        label=""
        options={DIAGNOSTIC_GROUPS}
        selected={surveyAnswers.diagnosticGroups}
        onChange={handleDiagnosticChange}
      />
    </div>
  );

  const renderVoice = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Voice Disorder</h1>
        <p className="text-slate-500 mt-2">Which voice condition(s) apply to you?</p>
      </div>
      <CheckboxGroup
        label=""
        options={VOICE_DISORDERS}
        selected={surveyAnswers.voiceDisorders}
        onChange={(v) => setSurveyAnswer("voiceDisorders", v)}
        toggleFn={(arr, val) => toggleDisorderArray(arr, val)}
      />
    </div>
  );

  const renderNeuro = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Neurological Disorder</h1>
        <p className="text-slate-500 mt-2">Which neurological condition(s) apply to you?</p>
      </div>
      <CheckboxGroup
        label=""
        options={NEURO_DISORDERS}
        selected={surveyAnswers.neurologicalDisorders}
        onChange={(v) => setSurveyAnswer("neurologicalDisorders", v)}
        toggleFn={(arr, val) => toggleDisorderArray(arr, val)}
      />
    </div>
  );

  const renderMood = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Mood & Psychiatric</h1>
        <p className="text-slate-500 mt-2">Which mood or psychiatric condition(s) apply to you?</p>
      </div>
      <CheckboxGroup
        label=""
        options={MOOD_DISORDERS}
        selected={surveyAnswers.moodDisorders}
        onChange={(v) => setSurveyAnswer("moodDisorders", v)}
        toggleFn={(arr, val) => toggleDisorderArray(arr, val)}
      />
    </div>
  );

  const renderRespiratory = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Respiratory Disorder</h1>
        <p className="text-slate-500 mt-2">Which respiratory condition(s) apply to you?</p>
      </div>
      <CheckboxGroup
        label=""
        options={RESPIRATORY_DISORDERS}
        selected={surveyAnswers.respiratoryDisorders}
        onChange={(v) => setSurveyAnswer("respiratoryDisorders", v)}
        toggleFn={(arr, val) => toggleDisorderArray(arr, val)}
      />
    </div>
  );

  const renderDemographics = () => (
    <div className="flex-1 space-y-6 mt-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Demographics</h1>
        <p className="text-slate-500 mt-2">Help us contextualize your biomarkers with a few baseline details.</p>
      </div>
      <div className="space-y-6">
        <RadioField
          label="Gender Identity"
          options={GENDER_OPTIONS}
          value={surveyAnswers.genderIdentity}
          onChange={(v) => setSurveyAnswer("genderIdentity", v)}
        />
        <RadioField
          label="Sexual Orientation"
          options={ORIENTATION_OPTIONS}
          value={surveyAnswers.sexualOrientation}
          onChange={(v) => setSurveyAnswer("sexualOrientation", v)}
        />
        <CheckboxGroup
          label="Race (select all that apply)"
          options={RACE_OPTIONS}
          selected={surveyAnswers.race}
          onChange={(v) => setSurveyAnswer("race", v)}
        />
        <RadioField
          label="Ethnicity"
          options={ETHNICITY_OPTIONS}
          value={surveyAnswers.ethnicity}
          onChange={(v) => setSurveyAnswer("ethnicity", v)}
        />
        <RadioField
          label="Primary Language"
          options={LANGUAGE_OPTIONS}
          value={surveyAnswers.primaryLanguage}
          onChange={(v) => setSurveyAnswer("primaryLanguage", v)}
        />
        <div className="space-y-3">
          <p className="text-sm font-medium text-slate-600">Age Group</p>
          <Select value={surveyAnswers.ageGroup} onValueChange={(v) => setSurveyAnswer("ageGroup", v)}>
            <SelectTrigger className="w-full bg-white border-violet-100 text-slate-700">
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
    <div className="flex flex-col h-full bg-[#faf8ff] text-slate-900 p-6 overflow-y-auto relative z-10">
      {/* Progress bar — hidden on consent step */}
      {currentStep !== "consent" && (
        <div className="mb-4 space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Step {currentStepIdx + 1} of {totalSteps}</span>
            <span>{Math.round(progressPercent)}%</span>
          </div>
          <Progress value={progressPercent} className="bg-violet-100 h-2" />
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

          {(currentStep !== "consent" || agreed) && (
          <div className="flex gap-3 mt-8">
            {currentStepIdx > 0 && (
              <button
                onClick={handleBack}
                className="flex-1 py-4 bg-violet-50 text-slate-700 rounded-2xl font-semibold hover:bg-violet-100 transition-colors flex justify-center items-center gap-2"
              >
                <ChevronLeft className="w-5 h-5" /> Back
              </button>
            )}
            <button
              onClick={handleNext}
              disabled={!canAdvance()}
              className="flex-1 py-4 bg-violet-600 text-white rounded-2xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-violet-700 transition-colors flex justify-center items-center gap-2 shadow-lg shadow-violet-300/30"
            >
              {isLastStep ? (
                <>Complete Calibration <CheckCircle2 className="w-5 h-5" /></>
              ) : (
                <>Continue <ChevronRight className="w-5 h-5" /></>
              )}
            </button>
          </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
