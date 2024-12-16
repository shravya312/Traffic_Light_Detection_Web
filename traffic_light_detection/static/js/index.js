document.getElementById('imageUpload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Send the image to Flask server
    const response = await fetch('/detect', {
        method: 'POST',
        body: formData,
    });

    // Parse the response from Flask (JSON with detected results)
    const results = await response.json();

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = () => {
        // Set canvas size to match the uploaded image
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw bounding boxes and labels
        results.forEach((result) => {
            const [x1, y1, x2, y2] = result.box;
            ctx.strokeStyle = getColor(result.light_color); // Traffic light color
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Display traffic light color and score
            ctx.fillStyle = getColor(result.light_color);
            ctx.font = '16px Arial';
            ctx.fillText(`${result.light_color} (${result.score.toFixed(2)})`, x1, y1 - 5);
        });
    };
    img.src = URL.createObjectURL(file);

    // Update status message
    document.getElementById('status').innerText = "Traffic lights detected!";
});

// Helper function to map color names to their corresponding color codes
function getColor(lightColor) {
    switch (lightColor.toLowerCase()) {
        case 'green':
            return 'green';
        case 'red':
            return 'red';
        case 'yellow':
            return 'yellow';
        default:
            return 'gray';
    }
}

// document.getElementById('imageUpload').addEventListener('change', async (event) => {
//     const file = event.target.files[0];
//     if (!file) return;

//     const formData = new FormData();
//     formData.append('file', file);

//     const response = await fetch('/detect', {
//         method: 'POST',
//         body: formData,
//     });

//     const results = await response.json();

//     const canvas = document.getElementById('canvas');
//     const ctx = canvas.getContext('2d');

//     const img = new Image();
//     img.onload = () => {
//         canvas.width = img.width;
//         canvas.height = img.height;
//         ctx.drawImage(img, 0, 0);

//         results.forEach((result) => {
//             const [x1, y1, x2, y2] = result.box;
//             ctx.strokeStyle = result.light_color.toLowerCase();
//             ctx.lineWidth = 3;
//             ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

//             ctx.fillStyle = result.light_color.toLowerCase();
//             ctx.font = '16px Arial';
//             ctx.fillText(`${result.light_color} (${result.score.toFixed(2)})`, x1, y1 - 5);
//         });
//     };
//     img.src = URL.createObjectURL(file);

//     document.getElementById('status').innerText = "Traffic lights detected!";
// });
