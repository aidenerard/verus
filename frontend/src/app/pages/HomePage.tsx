import { PAGE_CSS } from './home/tokens';
import Navbar       from './home/Navbar';
import Hero         from './home/Hero';
import WhyVerus     from './home/WhyVerus';
import TickerStripe from './home/TickerStripe';
import HowItWorks   from './home/HowItWorks';
import MethodSlider from './home/MethodSlider';
import OurPlatform  from './home/OurPlatform';
import Footer       from '../components/Footer';

export default function HomePage() {
  return (
    <div style={{ fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', overflowX: 'hidden' }}>
      <style>{PAGE_CSS}</style>
      <Navbar />
      <Hero />
      <WhyVerus />
      <TickerStripe />
      <HowItWorks />
      <MethodSlider />
      <OurPlatform />
      <Footer />
    </div>
  );
}
