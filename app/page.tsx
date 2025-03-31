'use client'

import { JSX, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Settings, HelpCircle, Mic, Video, StopCircle, Save } from "lucide-react";

export default function SignTranslateApp(): JSX.Element {
  const [isTranslating, setIsTranslating] = useState<boolean>(false);
  const [translationMode, setTranslationMode] = useState<"text" | "speech">("text");
  const [translatedText, setTranslatedText] = useState<string>("Translated text will appear here.");

  const handleTranslationToggle = (): void => setIsTranslating(!isTranslating);
  const handleModeSwitch = (): void => setTranslationMode(translationMode === "text" ? "speech" : "text");

  return (
    <div className="min-h-screen flex flex-col bg-gray-100 p-4">
      {/* Top Bar */}
      <div className="flex justify-between items-center bg-blue-500 text-white p-4 rounded-lg">
        <h1 className="text-xl font-bold">SignTranslate</h1>
        <div className="flex gap-4">
          <Settings className="cursor-pointer" />
          <HelpCircle className="cursor-pointer" />
        </div>
      </div>
      
      {/* Main Content */}
      <div className="flex flex-1 gap-4 mt-4">
        {/* Left Side - Video Feed */}
        <Card className={`flex-1 ${isTranslating ? "border-green-500 border-4" : ""}`}>
          <CardContent className="h-64 flex items-center justify-center text-gray-500 text-lg">
            {isTranslating ? <Video className="animate-pulse" size={64} /> : "Camera feed will appear here."}
          </CardContent>
        </Card>
        
        {/* Right Side - Translation Output */}
        <Card className="flex-1">
          <CardContent className="h-64 flex flex-col items-center justify-center text-xl">
            <p className="mb-2">{translatedText}</p>
            {translationMode === "speech" && <Mic className="animate-pulse" size={32} />}
          </CardContent>
        </Card>
      </div>
      
      {/* Bottom Bar */}
      <div className="flex justify-between items-center mt-4">
        <Button className={isTranslating ? "bg-red-500" : "bg-green-500"} onClick={handleTranslationToggle}>
          {isTranslating ? <StopCircle className="mr-2" /> : <Video className="mr-2" />} {isTranslating ? "Stop Translation" : "Start Translation"}
        </Button>
        <Button className="bg-blue-500" onClick={handleModeSwitch}>
          {translationMode === "text" ? "Switch to Speech Mode" : "Switch to Text Mode"}
        </Button>
        {isTranslating && <Button className="bg-yellow-500"><Save className="mr-2" /> Save Translation</Button>}
      </div>
    </div>
  );
}
