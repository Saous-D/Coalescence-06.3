$("#uploadButton").click(function() {
    if (imagePath === "") {
        alert("Please choose an image first");
        return;
    }

    $("#loading").show();
    $("#statusMessages").show().text("");

    var formData = new FormData();
    formData.append('image_path', imagePath);
    formData.append('image_height', localStorage.getItem('image_height') || '512');
    formData.append('image_width', localStorage.getItem('image_width') || '512');
    formData.append('water_penalty', localStorage.getItem('water_penalty') || '0.75');
    formData.append('no_water_penalty', localStorage.getItem('no_water_penalty') || '-0.75');
    formData.append('sigma', localStorage.getItem('sigma') || '0.3');
    formData.append('mu', localStorage.getItem('mu') || '1');

    const evtSource = new EventSource('/process');
    evtSource.onmessage = function(event) {
        if (event.data.startsWith('done: ')) {
            $("#loading").hide();
            evtSource.close();
            var processedImageData = "data:image/png;base64," + event.data.substring(6);
            $("#resultImage").attr("src", processedImageData).show();
            $("#downloadButton").attr("href", processedImageData).show();
        } else {
            $("#statusMessages").append(event.data + "<br>");
        }
    };
});
