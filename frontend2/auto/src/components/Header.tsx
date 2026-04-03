import { SignInButton, SignUpButton, UserButton, useAuth, Show } from "@clerk/react";
import { motion } from "framer-motion";

export const Header = () => {
  const { isSignedIn } = useAuth();

  return (
    <motion.header 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 glass-morphism"
    >
      <div className="flex items-center gap-3 group cursor-pointer">
        <div className="relative">
          <div className="absolute -inset-1 bg-primary/20 rounded-full blur-sm opacity-0 group-hover:opacity-100 transition-opacity" />
          <img 
            src="/image.png" 
            alt="AutoPilot ML Logo" 
            className="w-10 h-10 object-contain rounded-lg relative z-10" 
          />
        </div>
        <span className="text-xl font-black tracking-tighter text-gradient pb-1">AutoPilot ML</span>
      </div>

      <div className="flex items-center gap-4">
        <Show when="signed-out">
          <SignInButton mode="modal">
            <button className="px-4 py-2 text-sm font-medium transition-all hover:text-primary">
              Log In
            </button>
          </SignInButton>
          <SignUpButton mode="modal">
            <button className="px-5 py-2 text-sm font-medium text-white rounded-full bg-primary hover:shadow-glow transition-smooth">
              Get Started
            </button>
          </SignUpButton>
        </Show>
        <Show when="signed-in">
          <UserButton 
            appearance={{
              elements: {
                userButtonAvatarBox: "w-10 h-10 border border-white/20 shadow-lg"
              }
            }}
          />
        </Show>
      </div>
    </motion.header>
  );
};
