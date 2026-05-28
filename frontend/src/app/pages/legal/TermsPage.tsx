import { useEffect } from 'react';
import Navbar, { NAVBAR_HEIGHT } from '../../components/Navbar';
import Footer from '../../components/Footer';
import {
  LegalLayout, LegalHeader, Section, Subsection,
  MailLink, ContactCard, ArbitrationCallout, FootnoteDisclaimer,
} from './sections';
import TermsGeneral from './TermsGeneral';

export default function TermsPage() {
  useEffect(() => {
    window.scrollTo(0, 0);
    document.title = 'Terms of Use — Verus Technologies';
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#ffffff' }}>
      <Navbar />
      <div style={{ flex: 1, paddingTop: NAVBAR_HEIGHT }}>
        <LegalLayout>
          <LegalHeader
            eyebrow="Verus Technologies, Inc."
            title="Website Terms of Use"
            meta="Version 1.0 · Last Revised: May 28, 2026"
          />

          <p>
            The website located at <a href="https://www.try-verus.com" style={{ color: '#E8601C' }}>www.try-verus.com</a> (the "Site") is a copyrighted work belonging to Verus Technologies, Inc. ("Company", "us", "our", and "we"). Certain features of the Site may be subject to additional guidelines, terms, or rules, which will be posted on the Site in connection with such features. All such additional terms, guidelines, and rules are incorporated by reference into these Terms.
          </p>
          <p>
            These Terms of Use (these "Terms") set forth the legally binding terms and conditions that govern your use of the Site. By accessing or using the Site, you are accepting these Terms (on behalf of yourself or the entity that you represent), and you represent and warrant that you have the right, authority, and capacity to enter into these Terms (on behalf of yourself or the entity that you represent). You may not access or use the Site or accept the Terms if you are not at least 18 years old. If you do not agree with all of the provisions of these Terms, do not access and/or use the Site.
          </p>

          <ArbitrationCallout />

          <Section title="1. ACCOUNTS">
            <Subsection title="1.1 Account Creation">
              <p>In order to use certain features of the Site, you must register for an account ("Account") and provide certain information about yourself as prompted by the account registration form. You represent and warrant that: (a) all required registration information you submit is truthful and accurate; (b) you will maintain the accuracy of such information. You may delete your Account at any time, for any reason, by following the instructions on the Site. Company may suspend or terminate your Account in accordance with Section 8.</p>
            </Subsection>
            <Subsection title="1.2 Account Responsibilities">
              <p>You are responsible for maintaining the confidentiality of your Account login information and are fully responsible for all activities that occur under your Account. You agree to immediately notify Company of any unauthorized use, or suspected unauthorized use of your Account or any other breach of security. Company cannot and will not be liable for any loss or damage arising from your failure to comply with the above requirements.</p>
            </Subsection>
          </Section>

          <Section title="2. ACCESS TO THE SITE">
            <Subsection title="2.1 License">
              <p>Subject to these Terms, Company grants you a non-transferable, non-exclusive, revocable, limited license to use and access the Site solely for your own personal, noncommercial use.</p>
            </Subsection>
            <Subsection title="2.2 Certain Restrictions">
              <p>The rights granted to you in these Terms are subject to the following restrictions: (a) you shall not license, sell, rent, lease, transfer, assign, distribute, host, or otherwise commercially exploit the Site, whether in whole or in part, or any content displayed on the Site; (b) you shall not modify, make derivative works of, disassemble, reverse compile or reverse engineer any part of the Site; (c) you shall not access the Site in order to build a similar or competitive website, product, or service; and (d) except as expressly stated herein, no part of the Site may be copied, reproduced, distributed, republished, downloaded, displayed, posted or transmitted in any form or by any means. Unless otherwise indicated, any future release, update, or other addition to functionality of the Site shall be subject to these Terms. All copyright and other proprietary notices on the Site must be retained on all copies thereof.</p>
            </Subsection>
            <Subsection title="2.3 Modification">
              <p>Company reserves the right, at any time, to modify, suspend, or discontinue the Site (in whole or in part) with or without notice to you. You agree that Company will not be liable to you or to any third party for any modification, suspension, or discontinuation of the Site or any part thereof.</p>
            </Subsection>
            <Subsection title="2.4 No Support or Maintenance">
              <p>You acknowledge and agree that Company will have no obligation to provide you with any support or maintenance in connection with the Site.</p>
            </Subsection>
            <Subsection title="2.5 Ownership">
              <p>Excluding any User Content that you may provide, you acknowledge that all the intellectual property rights, including copyrights, patents, trademarks, and trade secrets, in the Site and its content are owned by Company or Company's suppliers. Neither these Terms nor your access to the Site transfers to you or any third party any rights, title or interest in or to such intellectual property rights, except for the limited access rights expressly set forth in Section 2.1. Company and its suppliers reserve all rights not granted in these Terms. There are no implied licenses granted under these Terms.</p>
            </Subsection>
            <Subsection title="2.6 Feedback">
              <p>If you provide Company with any feedback or suggestions regarding the Site ("Feedback"), you hereby assign to Company all rights in such Feedback and agree that Company shall have the right to use and fully exploit such Feedback and related information in any manner it deems appropriate. Company will treat any Feedback you provide to Company as non-confidential and non-proprietary. You agree that you will not submit to Company any information or ideas that you consider to be confidential or proprietary.</p>
            </Subsection>
          </Section>

          <Section title="3. USER CONTENT">
            <Subsection title="3.1 User Content">
              <p>"User Content" means any and all information and content that a user submits to, or uses with, the Site, including GPR data files uploaded for analysis. You are solely responsible for your User Content. You assume all risks associated with use of your User Content, including any reliance on its accuracy, completeness or usefulness by others. You hereby represent and warrant that your User Content does not violate our Acceptable Use Policy (defined in Section 3.3). Company is not obligated to backup any User Content, and your User Content may be deleted at any time without prior notice. You are solely responsible for creating and maintaining your own backup copies of your User Content if you desire.</p>
            </Subsection>
            <Subsection title="3.2 License">
              <p>You hereby grant to Company an irrevocable, nonexclusive, royalty-free and fully paid, worldwide license to reproduce, distribute, publicly display and perform, prepare derivative works of, incorporate into other works, and otherwise use and exploit your User Content, including uploaded GPR data files, solely for the purposes of providing the Service and training and improving Company's AI and machine learning models. You hereby irrevocably waive any claims and assertions of moral rights or attribution with respect to your User Content.</p>
            </Subsection>
            <Subsection title="3.3 Acceptable Use Policy">
              <p>You agree not to use the Site to collect, upload, transmit, display, or distribute any User Content (i) that violates any third-party right; (ii) that is unlawful, harassing, abusive, tortious, threatening, harmful, defamatory, false, or otherwise objectionable; (iii) that is harmful to minors in any way; or (iv) that is in violation of any law, regulation, or obligations or restrictions imposed by any third party.</p>
              <p>In addition, you agree not to: (i) upload computer viruses, worms, or any software intended to damage or alter a computer system or data; (ii) send unsolicited or unauthorized advertising or promotional materials; (iii) harvest or collect information regarding other users without their consent; (iv) interfere with, disrupt, or create an undue burden on servers or networks connected to the Site; (v) attempt to gain unauthorized access to the Site; (vi) harass or interfere with any other user's use and enjoyment of the Site; or (vii) use software or automated agents to produce multiple accounts or generate automated searches, requests, or queries to the Site.</p>
            </Subsection>
            <Subsection title="3.4 Enforcement">
              <p>We reserve the right to review, refuse and/or remove any User Content in our sole discretion, and to investigate and/or take appropriate action against you if you violate the Acceptable Use Policy or any other provision of these Terms. Such action may include removing or modifying your User Content, terminating your Account in accordance with Section 8, and/or reporting you to law enforcement authorities.</p>
            </Subsection>
          </Section>

          <Section title="4. INDEMNIFICATION">
            <p>You agree to indemnify and hold Company (and its officers, employees, and agents) harmless, including costs and attorneys' fees, from any claim or demand made by any third party due to or arising out of (a) your use of the Site, (b) your violation of these Terms, (c) your violation of applicable laws or regulations, or (d) your User Content. Company reserves the right, at your expense, to assume the exclusive defense and control of any matter for which you are required to indemnify us, and you agree to cooperate with our defense of these claims. You agree not to settle any matter without the prior written consent of Company. Company will use reasonable efforts to notify you of any such claim, action or proceeding upon becoming aware of it.</p>
          </Section>

          <Section title="5. THIRD-PARTY LINKS & ADS; OTHER USERS">
            <Subsection title="5.1 Third-Party Links & Ads">
              <p>The Site may contain links to third-party websites and services. Such third-party links are not under the control of Company, and Company is not responsible for any third-party links. Company provides access to these third-party links only as a convenience to you, and does not review, approve, monitor, endorse, warrant, or make any representations with respect to them. You use all third-party links at your own risk.</p>
            </Subsection>
            <Subsection title="5.2 Other Users">
              <p>Each Site user is solely responsible for any and all of its own User Content. Since we do not control User Content, you acknowledge and agree that we are not responsible for any User Content, whether provided by you or by others. Your interactions with other Site users are solely between you and such users. You agree that Company will not be responsible for any loss or damage incurred as the result of any such interactions.</p>
            </Subsection>
            <Subsection title="5.3 Release">
              <p>You hereby release and forever discharge Company (and our officers, employees, agents, successors, and assigns) from, and hereby waive and relinquish, each and every past, present and future dispute, claim, controversy, demand, right, obligation, liability, action and cause of action of every kind and nature that has arisen or arises directly or indirectly out of, or that relates directly or indirectly to, the Site. <span style={{ textTransform: 'uppercase' }}>If you are a California resident, you hereby waive California Civil Code Section 1542 in connection with the foregoing.</span></p>
            </Subsection>
          </Section>

          <Section title="6. DISCLAIMERS">
            <p style={{ textTransform: 'uppercase' }}>
              The Site is provided on an "as-is" and "as available" basis, and Company expressly disclaims any and all warranties and conditions of any kind, whether express, implied, or statutory, including all warranties or conditions of merchantability, fitness for a particular purpose, title, quiet enjoyment, accuracy, or non-infringement. We make no warranty that the Site will meet your requirements, will be available on an uninterrupted, timely, secure, or error-free basis, or will be accurate, reliable, free of viruses or other harmful code, complete, legal, or safe. If applicable law requires any warranties with respect to the Site, all such warranties are limited in duration to 90 days from the date of first use.
            </p>
          </Section>

          <Section title="7. LIMITATION ON LIABILITY">
            <p style={{ textTransform: 'uppercase' }}>
              To the maximum extent permitted by law, in no event shall Company be liable to you or any third party for any lost profits, lost data, costs of procurement of substitute products, or any indirect, consequential, exemplary, incidental, special or punitive damages arising from or relating to these Terms or your use of, or inability to use, the Site, even if Company has been advised of the possibility of such damages.
            </p>
            <p style={{ textTransform: 'uppercase' }}>
              To the maximum extent permitted by law, our liability to you for any damages arising from or related to these Terms will at all times be limited to a maximum of fifty US dollars ($50.00). The existence of more than one claim will not enlarge this limit.
            </p>
          </Section>

          <Section title="8. TERM AND TERMINATION">
            <p>Subject to this Section, these Terms will remain in full force and effect while you use the Site. We may suspend or terminate your rights to use the Site (including your Account) at any time for any reason at our sole discretion, including for any use of the Site in violation of these Terms. Upon termination of your rights under these Terms, your Account and right to access and use the Site will terminate immediately. Company will not have any liability whatsoever to you for any termination of your rights under these Terms, including for termination of your Account or deletion of your User Content. Even after your rights under these Terms are terminated, Sections 2.2 through 2.6, Section 3, and Sections 4 through 10 will remain in effect.</p>
          </Section>

          <Section title="9. COPYRIGHT POLICY">
            <p>Company respects the intellectual property of others and asks that users of our Site do the same. If you believe that one of our users is, through the use of our Site, unlawfully infringing the copyright(s) in a work, the following information must be provided to our designated Copyright Agent in writing pursuant to 17 U.S.C. § 512(c):</p>
            <ul style={{ paddingLeft: 24, marginBottom: 16 }}>
              <li style={{ marginBottom: 6 }}>Your physical or electronic signature;</li>
              <li style={{ marginBottom: 6 }}>Identification of the copyrighted work(s) that you claim to have been infringed;</li>
              <li style={{ marginBottom: 6 }}>Identification of the material on our services that you claim is infringing and that you request us to remove;</li>
              <li style={{ marginBottom: 6 }}>Sufficient information to permit us to locate such material;</li>
              <li style={{ marginBottom: 6 }}>Your address, telephone number, and e-mail address;</li>
              <li style={{ marginBottom: 6 }}>A statement that you have a good faith belief that use of the objectionable material is not authorized by the copyright owner, its agent, or under the law; and</li>
              <li style={{ marginBottom: 6 }}>A statement that the information in the notification is accurate, and under penalty of perjury, that you are either the owner of the copyright that has allegedly been infringed or that you are authorized to act on behalf of the copyright owner.</li>
            </ul>
            <ContactCard rows={[
              { label: '',                          value: <strong>Designated Copyright Agent: Aiden Erard</strong> },
              { label: 'Address',                   value: 'CREATE-X, Georgia Institute of Technology, 75 5th Street NW, Suite 314, Atlanta, GA 30308' },
              { label: 'Telephone',                 value: '(314) 885-4177' },
              { label: 'Email',                     value: <MailLink>info@verus.com</MailLink> },
            ]} />
          </Section>

          <TermsGeneral />

          <FootnoteDisclaimer>
            These Terms of Use were prepared using the Cooley GO template for general informational use. It is recommended that you seek legal advice from an appropriately licensed attorney to ensure these terms meet the unique requirements of your business and applicable law.
          </FootnoteDisclaimer>
        </LegalLayout>
      </div>
      <Footer />
    </div>
  );
}
