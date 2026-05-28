import { Section, Subsection, MailLink, ContactCard, LEGAL_ORANGE } from './sections';

export default function TermsGeneral() {
  return (
    <Section title="10. GENERAL">
      <Subsection title="10.1 Changes">
        <p>These Terms are subject to occasional revision. If we make any substantial changes, we may notify you by sending you an e-mail to the last e-mail address you provided to us, and/or by prominently posting notice of the changes on our Site. Continued use of our Site following notice of such changes shall indicate your acknowledgement of such changes and agreement to be bound by the terms and conditions of such changes.</p>
      </Subsection>

      <Subsection title="10.2 Dispute Resolution">
        <p><em>Please read the following arbitration agreement carefully. It requires you to arbitrate disputes with Company and limits the manner in which you can seek relief.</em></p>

        <p><strong>Applicability of Arbitration Agreement.</strong> You agree that any dispute between you and any of the Company Parties relating in any way to the Site, the services offered on the Site, or these Terms will be resolved by binding arbitration, rather than in court, except that (1) you and Company may assert individualized claims in small claims court if the claims qualify; and (2) you or Company may seek equitable relief in court for infringement or other misuse of intellectual property rights.</p>

        <p><strong>Informal Dispute Resolution.</strong> Before either party commences arbitration, the parties will personally meet and confer telephonically or via videoconference in a good faith effort to resolve the dispute informally. The party initiating a dispute must give written notice to the other party. Notice to Company should be sent to: <MailLink>info@verus.com</MailLink> or by regular mail to CREATE-X, Georgia Institute of Technology, 75 5th Street NW, Suite 314, Atlanta, GA 30308.</p>

        <p><strong>Arbitration Rules and Forum.</strong> If informal resolution fails within 60 days, either party may initiate binding arbitration through JAMS. The Federal Arbitration Act governs the interpretation and enforcement of this arbitration agreement. JAMS rules are available at www.jamsadr.com or by calling 800-352-5267.</p>

        <p style={{ textTransform: 'uppercase' }}>
          <strong>Waiver of Jury Trial.</strong> Except as specified in Section 10.2, you and Company hereby waive any constitutional and statutory rights to sue in court and have a trial in front of a judge or a jury. All covered claims and disputes shall be resolved exclusively by arbitration under this arbitration agreement.
        </p>

        <p style={{ textTransform: 'uppercase' }}>
          <strong>Waiver of Class or Other Non-Individualized Relief.</strong> You and Company agree that each of us may bring claims against the other only on an individual basis and not on a class, representative, or collective basis. Only individual relief is available, and disputes of more than one customer or user cannot be arbitrated or consolidated with those of any other customer or user.
        </p>

        <p><strong>30-Day Right to Opt Out.</strong> You have the right to opt out of the provisions of this arbitration agreement by sending written notice to: CREATE-X, Georgia Institute of Technology, 75 5th Street NW, Suite 314, Atlanta, GA 30308, or email to <MailLink>info@verus.com</MailLink>, within 30 days after first becoming subject to this arbitration agreement.</p>
      </Subsection>

      <Subsection title="10.3 Export">
        <p>The Site may be subject to U.S. export control laws. You agree not to export, reexport, or transfer, directly or indirectly, any U.S. technical data acquired from Company in violation of the United States export laws or regulations.</p>
      </Subsection>

      <Subsection title="10.4 Disclosures">
        <p>If you are a California resident, you may report complaints to the Complaint Assistance Unit of the Division of Consumer Product of the California Department of Consumer Affairs by contacting them in writing at 400 R Street, Sacramento, CA 95814, or by telephone at (800) 952-5210.</p>
      </Subsection>

      <Subsection title="10.5 Electronic Communications">
        <p>You consent to receive communications from Company in an electronic form and agree that all terms and conditions, agreements, notices, disclosures, and other communications that Company provides to you electronically satisfy any legal requirement that such communications would satisfy if in hardcopy writing.</p>
      </Subsection>

      <Subsection title="10.6 Entire Terms">
        <p>These Terms constitute the entire agreement between you and us regarding the use of the Site. If any provision of these Terms is held to be invalid or unenforceable, the other provisions of these Terms will be unimpaired and the invalid or unenforceable provision will be deemed modified so that it is valid and enforceable to the maximum extent permitted by law.</p>
      </Subsection>

      <Subsection title="10.7 Copyright/Trademark Information">
        <p>Copyright © 2026 Verus Technologies, Inc. All rights reserved. All trademarks, logos and service marks displayed on the Site are our property or the property of other third parties. You are not permitted to use these marks without our prior written consent.</p>
      </Subsection>

      <Subsection title="10.8 Contact Information">
        <ContactCard rows={[
          { label: '',         value: <strong>Verus Technologies, Inc.</strong> },
          { label: 'Address',   value: 'CREATE-X, Georgia Institute of Technology, 75 5th Street NW, Suite 314, Atlanta, GA 30308' },
          { label: 'Telephone', value: '(314) 885-4177' },
          { label: 'Email',     value: <MailLink>info@verus.com</MailLink> },
          { label: 'Website',   value: <a href="https://try-verus.com" style={{ color: LEGAL_ORANGE }}>try-verus.com</a> },
        ]} />
      </Subsection>
    </Section>
  );
}
