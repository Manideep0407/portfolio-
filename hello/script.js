// Helper function to show the certificate
function showCertificate(index, sliderId) {
    const container = document.querySelector(`#${sliderId}`);
    const certificates = container.querySelectorAll(".certification");

    certificates.forEach((certificate, i) => {
        certificate.classList.remove("active");
        if (i === index) {
            certificate.classList.add("active");
        }
    });
}

// Function to handle certificate change
function changeCertificate(direction, sliderId) {
    const container = document.querySelector(`#${sliderId}`);
    const certificates = container.querySelectorAll(".certification");

    let currentIndex = Array.from(certificates).findIndex(cert =>
        cert.classList.contains("active")
    );

    currentIndex += direction;

    if (currentIndex < 0) currentIndex = certificates.length - 1;
    if (currentIndex >= certificates.length) currentIndex = 0;

    showCertificate(currentIndex, sliderId);
}

// Auto-switch every 3 seconds (optional)
setInterval(() => {
    changeCertificate(1, "my-certificates");
    changeCertificate(1, "extra-certificates");
}, 3000);
link.href = "resume/Manideep.Nellipalli.pdf";
