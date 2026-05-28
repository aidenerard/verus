import type { ReactNode } from 'react';

export const LEGAL_ORANGE = '#E8601C';

export function LegalLayout({ children }: { children: ReactNode }) {
  return (
    <div style={{
      maxWidth: 760,
      margin: '0 auto',
      padding: '60px 24px 80px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
      color: '#1a1a1a',
      lineHeight: 1.7,
    }}>
      {children}
    </div>
  );
}

interface LegalHeaderProps {
  eyebrow:    string;
  title:      string;
  meta:       string;
}

export function LegalHeader({ eyebrow, title, meta }: LegalHeaderProps) {
  return (
    <div style={{ marginBottom: 40, borderBottom: `2px solid ${LEGAL_ORANGE}`, paddingBottom: 24 }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.1em', color: LEGAL_ORANGE, textTransform: 'uppercase', marginBottom: 8 }}>
        {eyebrow}
      </div>
      <h1 style={{ fontSize: 32, fontWeight: 700, margin: '0 0 8px', color: '#111' }}>{title}</h1>
      <div style={{ fontSize: 13, color: '#6b7280' }}>{meta}</div>
    </div>
  );
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginTop: 40 }}>
      <h2 style={{
        fontSize: 18, fontWeight: 700, color: '#111',
        borderBottom: '1px solid #e5e7eb', paddingBottom: 10,
        marginBottom: 16,
      }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

export function Subsection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginTop: 20 }}>
      <h3 style={{ fontSize: 15, fontWeight: 600, color: '#374151', marginBottom: 10 }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

export function MailLink({ children }: { children: ReactNode }) {
  return <a href="mailto:info@verus.com" style={{ color: LEGAL_ORANGE }}>{children}</a>;
}

interface ContactCardProps { rows: { label: string; value: ReactNode }[] }

export function ContactCard({ rows }: ContactCardProps) {
  return (
    <div style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 8, padding: 20, marginTop: 12 }}>
      {rows.map((r, i) => (
        <p key={r.label} style={{ margin: i === 0 ? '0 0 4px' : '0 0 4px' }}>
          {r.label === '' ? null : <><strong>{r.label}: </strong></>}{r.value}
        </p>
      ))}
    </div>
  );
}

export function FootnoteDisclaimer({ children }: { children: ReactNode }) {
  return (
    <div style={{ marginTop: 48, paddingTop: 24, borderTop: '1px solid #e5e7eb', fontSize: 12, color: '#9ca3af', fontStyle: 'italic' }}>
      {children}
    </div>
  );
}

export function ArbitrationCallout() {
  return (
    <div style={{
      background: '#fff8f6',
      border: `2px solid ${LEGAL_ORANGE}`,
      borderRadius: 8,
      padding: 20,
      margin: '24px 0',
      fontSize: 13,
      lineHeight: 1.6,
    }}>
      <p style={{ fontWeight: 700, margin: '0 0 8px' }}>IMPORTANT NOTICE REGARDING ARBITRATION</p>
      <p style={{ margin: '0 0 8px' }}>
        PLEASE BE AWARE THAT SECTION 10.2 CONTAINS PROVISIONS GOVERNING HOW TO RESOLVE DISPUTES BETWEEN YOU AND COMPANY. AMONG OTHER THINGS, SECTION 10.2 INCLUDES AN AGREEMENT TO ARBITRATE WHICH REQUIRES, WITH LIMITED EXCEPTIONS, THAT ALL DISPUTES BETWEEN YOU AND US SHALL BE RESOLVED BY BINDING AND FINAL ARBITRATION. SECTION 10.2 ALSO CONTAINS A CLASS ACTION AND JURY TRIAL WAIVER. PLEASE READ SECTION 10.2 CAREFULLY.
      </p>
      <p style={{ margin: 0 }}>
        UNLESS YOU OPT OUT OF THE AGREEMENT TO ARBITRATE WITHIN 30 DAYS: (1) YOU WILL ONLY BE PERMITTED TO PURSUE DISPUTES OR CLAIMS AND SEEK RELIEF AGAINST US ON AN INDIVIDUAL BASIS, NOT AS A PLAINTIFF OR CLASS MEMBER IN ANY CLASS OR REPRESENTATIVE ACTION OR PROCEEDING AND YOU WAIVE YOUR RIGHT TO PARTICIPATE IN A CLASS ACTION LAWSUIT OR CLASS-WIDE ARBITRATION; AND (2) YOU ARE WAIVING YOUR RIGHT TO PURSUE DISPUTES OR CLAIMS AND SEEK RELIEF IN A COURT OF LAW AND TO HAVE A JURY TRIAL.
      </p>
    </div>
  );
}

export const ALL_CAPS_PARA: React.CSSProperties = {
  textTransform: 'uppercase' as const,
};
