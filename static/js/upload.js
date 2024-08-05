document.getElementById('lp-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch('/process_lp', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultElement = document.getElementById('result');
        const resultImageContainer = document.getElementById('result-image');
        const submittedLPElement = document.getElementById('submitted-lp');
        const basisResultElement = document.getElementById('basis-result');
        const basisElement = document.getElementById('basis');

        // Clear previous image if it exists
        resultImageContainer.innerHTML = '';

        if (data.error) {
            resultElement.textContent = `Error: ${data.error}`;
            submittedLPElement.textContent = '';
            basisResultElement.style.display = 'none';
        } else {
            submittedLPElement.textContent = `A: ${JSON.stringify(data['A'], null, 2)}\n\nb: ${JSON.stringify(data['b'], null, 2)}\n\nc: ${JSON.stringify(data['c'], null, 2)}`;
            basisElement.textContent = `Accuracy: ${data['Accuracy']}\n\nSIB Basis:\n${data['SIB Basis']}\n\nOptimal Basis:\n${data['Optimal Basis']}`;

            if (data.image_base64) {
                const imageElement = document.createElement('img');
                imageElement.src = `data:image/png;base64,${data.image_base64}`;
                imageElement.alt = 'LP Problem Visualization';
                resultImageContainer.appendChild(imageElement);
            }

            // Display the basis result section
            basisResultElement.style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'Error occurred while processing the file.';
    });
});