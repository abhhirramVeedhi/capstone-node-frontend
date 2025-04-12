// const video = document.getElementById("camera");
// const canvas = document.getElementById("snapshot");

// navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
//   video.srcObject = stream;
// });

// function capture() {
//   canvas.width = video.videoWidth;
//   canvas.height = video.videoHeight;
//   canvas.getContext("2d").drawImage(video, 0, 0);
//   const imageData = canvas.toDataURL("image/png");
//   // TODO: send imageData to backend
// }