import { useEffect } from 'react';
import Footer from '../../components/Footer';
import {
  LegalLayout, LegalHeader, Section, Subsection,
  MailLink, ContactCard, FootnoteDisclaimer,
} from './sections';

export default function PrivacyPage() {
  useEffect(() => {
    window.scrollTo(0, 0);
    document.title = 'Privacy Policy — Verus Technologies';
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#ffffff' }}>
      <div style={{ flex: 1 }}>
        <LegalLayout>
          <LegalHeader
            eyebrow="Verus Technologies, Inc."
            title="Privacy Policy"
            meta="Last Updated: May 28, 2026"
          />

          <p>
            Verus Technologies, Inc. (the "Company") is committed to maintaining robust privacy protections for its users. Our Privacy Policy ("Privacy Policy") is designed to help you understand how we collect, use, and safeguard the information you provide to us and to assist you in making informed decisions when using our Service.
          </p>
          <p>
            For purposes of this Agreement, <strong>"Site"</strong> refers to the Company's website, which can be accessed at try-verus.com. <strong>"Service"</strong> refers to the Company's AI-powered GPR (ground-penetrating radar) inspection analysis platform, accessed via the Site, through which users upload subsurface sensor data files and receive automated condition maps and structural analysis reports. The terms <strong>"we," "us,"</strong> and <strong>"our"</strong> refer to Verus Technologies, Inc. <strong>"You"</strong> refers to you, as a user of our Site or our Service.
          </p>
          <p>
            By accessing our Site or our Service, you accept our Privacy Policy and consent to our collection, storage, use, and disclosure of your Personal Information as described in this Privacy Policy.
          </p>

          <Section title="I. Information We Collect">
            <p>We collect <strong>"Non-Personal Information"</strong> and <strong>"Personal Information."</strong></p>
            <p>Non-Personal Information includes information that cannot be used to personally identify you, such as anonymous usage data, browser type, device type, referring URLs, and number of page visits.</p>
            <p>Personal Information includes your name, email address, and company or agency name, which you submit to us through the registration process on the Site.</p>

            <Subsection title="1. Information Collected via Technology">
              <p>To activate the Service, you do not need to submit any Personal Information other than your email address, name, and company or agency name. In an effort to improve the quality of the Service, we track information provided to us by your browser when you view or use the Service, such as the website you came from (known as the "referring URL"), the type of browser you use, the device from which you connected to the Service, the time and date of access, and other information that does not personally identify you. We track this information using cookies, or small text files which include an anonymous unique identifier. Cookies are sent to a user's browser from our servers and are stored on the user's computer hard drive. Sending a cookie to a user's browser enables us to collect Non-Personal Information about that user and keep a record of the user's preferences when utilizing our services, both on an individual and aggregate basis.</p>
              <p>The Company uses both persistent and session cookies; persistent cookies remain on your computer after you close your session and until you delete them, while session cookies expire when you close your browser.</p>
            </Subsection>

            <Subsection title="2. Information You Provide by Registering for an Account">
              <p>To use the Service, you will need to create an account. You can create an account by registering with the Service and entering your name, email address, company or agency name, and creating a password. By registering, you are authorizing us to collect, store, and use your information in accordance with this Privacy Policy.</p>
            </Subsection>

            <Subsection title="3. Uploaded Data">
              <p>The Service allows you to upload GPR sensor data files (including but not limited to .dzt and .xlsx formats) for analysis. These uploaded data files are stored indefinitely on our secure servers hosted by Supabase. We use uploaded data to provide analysis results and to train and improve our AI and machine learning models. By uploading data to the Service, you grant Verus Technologies, Inc. a non-exclusive, worldwide, royalty-free license to store, process, and use that data for the purposes described in this Privacy Policy.</p>
              <p>If you wish to request deletion of your uploaded data, please contact us at <MailLink>info@verus.com</MailLink>.</p>
            </Subsection>

            <Subsection title="4. Children's Privacy">
              <p>The Site and the Service are not directed to anyone under the age of 13. The Site does not knowingly collect or solicit information from anyone under the age of 13, or allow anyone under the age of 13 to sign up for the Service. In the event that we learn that we have gathered personal information from anyone under the age of 13 without the consent of a parent or guardian, we will delete that information as soon as possible. If you believe we have collected such information, please contact us at <MailLink>info@verus.com</MailLink>.</p>
            </Subsection>
          </Section>

          <Section title="II. How We Use and Share Information">
            <p><strong>Personal Information:</strong></p>
            <p>Except as otherwise stated in this Privacy Policy, we do not sell, trade, rent, or otherwise share for marketing purposes your Personal Information with third parties without your consent. We do share Personal Information with vendors who are performing services for the Company, including:</p>
            <ul style={{ paddingLeft: 24, marginBottom: 16 }}>
              <li style={{ marginBottom: 8 }}><strong>Supabase, Inc.</strong> — database hosting, file storage, and user authentication. Supabase stores your account information and uploaded GPR data files on our behalf.</li>
              <li style={{ marginBottom: 8 }}><strong>Render Technologies, Inc.</strong> — backend server hosting. Render processes inference requests when you run an analysis through the Service.</li>
              <li style={{ marginBottom: 8 }}><strong>Vercel, Inc.</strong> — frontend hosting. Vercel serves the Site to your browser.</li>
            </ul>
            <p>These vendors use your Personal Information only at our direction and in accordance with our Privacy Policy. We use Personal Information to contact users in response to questions, solicit feedback, provide technical support, and inform users about product updates.</p>
            <p>We may share Personal Information with outside parties if we have a good-faith belief that access, use, preservation, or disclosure of the information is reasonably necessary to meet any applicable legal process or enforceable governmental request; to enforce applicable Terms of Service, including investigation of potential violations; address fraud, security, or technical concerns; or to protect against harm to the rights, property, or safety of our users or the public as required or permitted by law.</p>
            <p><strong>Non-Personal Information:</strong></p>
            <p>In general, we use Non-Personal Information to help us improve the Service and customize the user experience. We also aggregate Non-Personal Information in order to track trends and analyze use patterns on the Site. This Privacy Policy does not limit in any way our use or disclosure of Non-Personal Information and we reserve the right to use and disclose such Non-Personal Information to our partners and other third parties at our discretion.</p>
            <p><strong>Business Transactions:</strong></p>
            <p>In the event we undergo a business transaction such as a merger, acquisition by another company, or sale of all or a portion of our assets, your Personal Information and uploaded data may be among the assets transferred. You acknowledge and consent that such transfers may occur and are permitted by this Privacy Policy, and that any acquirer of our assets may continue to process your Personal Information and data as set forth in this Privacy Policy.</p>
          </Section>

          <Section title="III. How We Protect Information">
            <p>We implement security measures designed to protect your information from unauthorized access. Your account is protected by your account password and we urge you to take steps to keep your personal information safe by not disclosing your password and by logging out of your account after each use. We further protect your information from potential security breaches by implementing technological security measures including encryption, firewalls, and secure socket layer technology through our hosting providers Supabase, Render, and Vercel.</p>
            <p>However, these measures do not guarantee that your information will not be accessed, disclosed, altered, or destroyed by breach of such firewalls and secure server software. By using our Service, you acknowledge that you understand and agree to assume these risks.</p>
          </Section>

          <Section title="IV. Your Rights Regarding the Use of Your Personal Information">
            <p>You have the right at any time to prevent us from contacting you for marketing purposes. When we send a promotional communication to a user, the user can opt out of further promotional communications by following the unsubscribe instructions provided in each promotional email.</p>
            <p>You may also request access to, correction of, or deletion of your Personal Information or uploaded data at any time by contacting us at <MailLink>info@verus.com</MailLink>. We will respond to such requests within a reasonable timeframe.</p>
          </Section>

          <Section title="V. Links to Other Websites">
            <p>As part of the Service, we may provide links to or compatibility with other websites or applications. However, we are not responsible for the privacy practices employed by those websites or the information or content they contain. This Privacy Policy applies solely to information collected by us through the Site and the Service. We encourage our users to read the privacy statements of other websites before proceeding to use them.</p>
          </Section>

          <Section title="VI. Changes to Our Privacy Policy">
            <p>Verus Technologies, Inc. reserves the right to change this policy at any time. We will notify you of significant changes to our Privacy Policy by sending a notice to the primary email address specified in your account or by placing a prominent notice on our Site. Significant changes will go into effect 30 days following such notification. Non-material changes or clarifications will take effect immediately. You should periodically check the Site and this privacy page for updates.</p>
          </Section>

          <Section title="VII. Contact Us">
            <p>If you have any questions regarding this Privacy Policy or the practices of this Site, please contact us:</p>
            <ContactCard rows={[
              { label: '', value: <strong>Verus Technologies, Inc.</strong> },
              { label: 'Email',   value: <MailLink>info@verus.com</MailLink> },
              { label: 'Website', value: <a href="https://try-verus.com" style={{ color: '#E8601C' }}>try-verus.com</a> },
            ]} />
          </Section>

          <FootnoteDisclaimer>
            This Privacy Policy was prepared for general informational use. It is recommended that you seek legal advice from an appropriately licensed attorney to ensure this policy meets the unique requirements of your business and applicable law.
          </FootnoteDisclaimer>
        </LegalLayout>
      </div>
      <Footer />
    </div>
  );
}
