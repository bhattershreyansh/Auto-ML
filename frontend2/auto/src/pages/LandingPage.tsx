import { motion } from "framer-motion";
import { 
  Rocket, 
  ShieldCheck, 
  Zap, 
  LineChart, 
  Search, 
  CloudLightning,
  ChevronRight
} from "lucide-react";
import { SignInButton } from "@clerk/react";
import { Header } from "@/components/Header";

const FeatureCard = ({ icon: Icon, title, description, delay }: { icon: any, title: string, description: string, delay: number }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    viewport={{ once: true }}
    className="p-8 glass-card group hover:scale-[1.02] transition-all duration-500"
  >
    <div className="w-12 h-12 p-3 mb-6 rounded-2xl bg-primary/10 border border-primary/20 group-hover:bg-primary/20 transition-colors">
      <Icon className="w-full h-full text-primary" />
    </div>
    <h3 className="mb-3 text-xl font-bold text-white">{title}</h3>
    <p className="text-slate-400 leading-relaxed">{description}</p>
  </motion.div>
);

const LandingPage = () => {
  return (
    <div className="min-h-screen text-white mesh-gradient selection:bg-primary/30">
      <Header />
      
      {/* Hero Section */}
      <section className="relative flex flex-col items-center justify-center min-h-screen px-6 pt-20 text-center overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-primary/10 rounded-full blur-[120px]" />
          <div className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-purple-500/10 rounded-full blur-[100px]" />
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 max-w-5xl"
        >
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 mb-8 text-sm font-medium border rounded-full glass-morphism border-white/10"
          >
            <span className="flex w-2 h-2 rounded-full bg-primary shadow-glow animate-pulse" />
            <span className="text-slate-400">Next-Gen AutoML Engine</span>
          </motion.div>

          <h1 className="mb-8 text-6xl font-black md:text-8xl tracking-tighter leading-none">
            <span className="block text-gradient">Master Your Data.</span>
            <span className="block opacity-80">Own the Model.</span>
          </h1>

          <p className="max-w-2xl mx-auto mb-10 text-xl font-medium leading-relaxed text-slate-400">
            AutoPilot ML is a high-performance workspace to upload datasets, select the best algorithms, 
            and deploy production-ready models in seconds. Fully secure and multi-tenant.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-6">
            <SignInButton mode="modal">
              <button className="flex items-center gap-2 px-10 py-5 text-lg font-bold text-white transition-all rounded-2xl bg-primary hover:shadow-glow hover:-translate-y-1">
                Start Training Now
                <ChevronRight className="w-6 h-6" />
              </button>
            </SignInButton>
            <button className="px-10 py-5 text-lg font-bold transition-all border rounded-2xl glass-morphism border-white/10 hover:bg-white/5">
              Explore Documentation
            </button>
          </div>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 px-6 py-32 mx-auto max-w-7xl">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
          <FeatureCard 
            icon={Zap}
            delay={0.1}
            title="Insight Engine"
            description="Go beyond metrics. Use SHAP explainability to understand exactly why your models make specific predictions."
          />
          <FeatureCard 
            icon={ShieldCheck}
            delay={0.2}
            title="Drift Sentry"
            description="Detect data drift before it hits production. Monitor your distribution shifts with automated alerts."
          />
          <FeatureCard 
            icon={CloudLightning}
            delay={0.3}
            title="Instant Deploy"
            description="Convert your trained models into ready-to-use FastAPI endpoints with a single click. Zero boilerplate."
          />
          <FeatureCard 
            icon={LineChart}
            delay={0.4}
            title="History Engine"
            description="A complete audit trail of every experiment. Compare metrics side-by-side and roll back with ease."
          />
          <FeatureCard 
            icon={Search}
            delay={0.5}
            title="Secure Isolation"
            description="Your data is yours. Every tenant gets a private, encrypted sandboxed environment for their datasets."
          />
          <FeatureCard 
            icon={Rocket}
            delay={0.6}
            title="One-Click AutoML"
            description="From raw CSV to optimized Random Forest, XGBoost, or LightGBM in under 60 seconds."
          />
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 px-6 border-t border-white/10 text-center">
        <p className="text-slate-500 font-medium">
          &copy; {new Date().getFullYear()} AutoPilot ML. Built for high-performance ML teams.
        </p>
      </footer>
    </div>
  );
};

export default LandingPage;
