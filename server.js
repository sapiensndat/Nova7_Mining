const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

const transporter = nodemailer.createTransporter({
    service: 'outlook', // Updated to Outlook for Microsoft email
    auth: {
        user: 'sapiens@xeltis.org', // Your email
        pass: 'Godisfortue25.' // Replace with your Outlook password or app-specific password
    }
});

app.post('/api/send-help-email', async (req, res) => {
    const { name, phone, email, wantTrial, issue, to } = req.body;

    const mailOptions = {
        from: 'sapiens@xeltis.org', // Your email
        to: to,
        replyTo: email,
        subject: `Help Request from ${name}${wantTrial ? ' (Trial Requested)' : ''}`,
        text: `
            Name: ${name}
            Phone: ${phone}
            Email: ${email}
            Want a Trial: ${wantTrial ? 'Yes' : 'No'}
            Issue: ${issue}
        `
    };

    try {
        await transporter.sendMail(mailOptions);
        res.status(200).json({ message: 'Email sent successfully' });
    } catch (error) {
        console.error('Error sending email:', error);
        res.status(500).json({ error: 'Failed to send email' });
    }
});

app.listen(5000, () => {
    console.log('Server running on port 5000');
});