socket = io()
document.getElementById("upload_button").addEventListener("click", function () {
    file_uploader(socket)
})
document.getElementById("predict_button").addEventListener("click", function () {
    predict_plant(socket)
})

function predict_plant(socket){
    socket.emit("predict");
}


function file_uploader(socket) {
    var file_input = document.getElementById("file_selector");
    if (file_input.files.length>0){
        image = file_input.files[0]
        
        socket.emit("file_upload",image);
    }
    else{
        console.log("no file")
    }
}
socket.on("prediction",function(data){
    document.getElementById("prediction").textContent = data.data;
});

function display_image(event){
    var file_input = document.getElementById("file_selector");
    var file_display = document.getElementById("flower_image");
    if (file_input.files.length>0){
        image = file_input.files[0]
        file_display.src = URL.createObjectURL(image);
    }
    else{
        console.log("no file")
    }
}
