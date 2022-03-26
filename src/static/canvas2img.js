var canvas = document.getElementById("canvas")
var ctx = canvas.getContext('2d')

var drawing = false;
var before_x = 0;
var before_y = 0;

// キャンバスの描画処理イベント追加
canvas.addEventListener('mousemove', draw_canvas);

canvas.addEventListener('mousedown', function(e){
    drawing = true;

    var mouse_point = e.target.getBoundingClientRect();
    before_x = e.clientX - mouse_point.left;
    before_y = e.clientY - mouse_point.top;
});

canvas.addEventListener('mouseup', function(e){
    drawing = false;
});

// キャンバスへの描画処理
function draw_canvas(e){

    if (!drawing) return;

    var mouse_point = e.target.getBoundingClientRect();
    x = e.clientX - mouse_point.left;
    y = e.clientY - mouse_point.top;

    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 10;

    ctx.beginPath();
    ctx.moveTo(before_x, before_y);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.closePath();
    
    before_x = x;
    before_y = y;
}

// キャンバスの初期化（クリア）処理
document.getElementById('btn-clear').addEventListener("click", function(e){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// キャンバスの画像をサーバーに送るイベントを追加
document.getElementById('btn-send').addEventListener("click", function(e){
    const image = document.getElementById("canvas").toDataURL("image/png")
    const predict_url = "/predict"

    const param = {
        method : "POST",
        headers : {
            "Content-Type": "application/json; charset=utf-8"
        },
        body : JSON.stringify({data : image})
    };

    sendServer(predict_url, param);
});

// キャンバスの画像をサーバーに送る処理
function sendServer(url, param){
    fetch(url, param)
    .then(function(response){
        return response.json();
    })
    .then(function(json){
        if(typeof json.predict === 'undefined'){
            alert("Fault 1.");
        }else{
            console.log(json.predict)
            // 表の書き換え部分
            for (i = 0; i < 10; i++){
                var result_tab = document.getElementById("result-table")
                result_tab.rows[1].cells[i+1].innerText = json.predict[i];
            }
        }
    })
    .catch(function(error){
        alert("Fault 2.");
        console.log(error);
    });
}