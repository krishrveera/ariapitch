import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type SurveyAnswers = {
  diagnosticGroups: string[];
  voiceDisorders: string[];
  neurologicalDisorders: string[];
  moodDisorders: string[];
  respiratoryDisorders: string[];
  genderIdentity: string;
  sexualOrientation: string;
  race: string[];
  ethnicity: string;
  primaryLanguage: string;
  ageGroup: string;
};

const defaultSurveyAnswers: SurveyAnswers = {
  diagnosticGroups: [],
  voiceDisorders: [],
  neurologicalDisorders: [],
  moodDisorders: [],
  respiratoryDisorders: [],
  genderIdentity: '',
  sexualOrientation: '',
  race: [],
  ethnicity: '',
  primaryLanguage: '',
  ageGroup: '',
};

export type AudioRecording = {
  id: string;
  date: string;
  timestamp: number;
  pitch: number;
  shimmer: number;
  jitter: number;
  message: string;
  isAnomaly: boolean;
  duration: number;
  audioData?: Float32Array;
  sampleRate?: number;
  spectralCentroid?: number;
  harmonicRatio?: number;
  formants?: number[];
};

type UserData = {
  optedIn: boolean;
  onboardingComplete: boolean;
  demographics: any;
  hasRecordedToday: boolean;
  streak: number;
  history: Array<{
    date: string;
    pitch: number;
    shimmer: number;
    jitter: number;
    spectralCentroid?: number;
    harmonicRatio?: number;
    message: string;
    isAnomaly: boolean;
  }>;
  audioRecordings: AudioRecording[];
  showHealthPopup: boolean;
  lastRecordingData?: Float32Array;
  lastRecordingSampleRate?: number;
};

const defaultData: UserData = {
  optedIn: false,
  onboardingComplete: false,
  demographics: null,
  hasRecordedToday: false,
  streak: 5,
  history: [
    { date: 'Monday, Mar 3', pitch: 210, shimmer: 3.2, jitter: 1.1, spectralCentroid: 1480, harmonicRatio: 17.2, message: "Your voice sounds steady today. Great job!", isAnomaly: false },
    { date: 'Tuesday, Mar 4', pitch: 215, shimmer: 3.5, jitter: 1.2, spectralCentroid: 1520, harmonicRatio: 18.1, message: "Slight variation in shimmer, but within normal range.", isAnomaly: false },
    { date: 'Wednesday, Mar 5', pitch: 208, shimmer: 3.1, jitter: 1.0, spectralCentroid: 1510, harmonicRatio: 19.3, message: "Looking perfectly healthy!", isAnomaly: false },
    { date: 'Thursday, Mar 6', pitch: 212, shimmer: 3.8, jitter: 1.3, spectralCentroid: 1490, harmonicRatio: 16.8, message: "Pitch is right on target.", isAnomaly: false },
    { date: 'Friday, Mar 7', pitch: 210, shimmer: 3.4, jitter: 1.1, spectralCentroid: 1530, harmonicRatio: 18.5, message: "Voice is clear. Keep hydrating!", isAnomaly: false },
    { date: 'Saturday, Mar 8', pitch: 218, shimmer: 3.9, jitter: 1.4, spectralCentroid: 1545, harmonicRatio: 17.9, message: "A little extra effort detected, remember to rest.", isAnomaly: false },
  ],
  audioRecordings: [],
  showHealthPopup: false,
};

type AppContextType = {
  userData: UserData;
  setUserData: React.Dispatch<React.SetStateAction<UserData>>;
  surveyAnswers: SurveyAnswers;
  setSurveyAnswer: <K extends keyof SurveyAnswers>(key: K, value: SurveyAnswers[K]) => void;
};

const AppContext = createContext<AppContextType | null>(null);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [userData, setUserData] = useState<UserData>(defaultData);
  const [surveyAnswers, setSurveyAnswers] = useState<SurveyAnswers>(defaultSurveyAnswers);

  const setSurveyAnswer = useCallback(<K extends keyof SurveyAnswers>(key: K, value: SurveyAnswers[K]) => {
    setSurveyAnswers(prev => ({ ...prev, [key]: value }));
  }, []);

  return (
    <AppContext.Provider value={{ userData, setUserData, surveyAnswers, setSurveyAnswer }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("Missing provider");
  return ctx;
};
