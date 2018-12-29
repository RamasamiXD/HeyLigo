let length = 1000;
let stars = new Array();
let speed;
function windowResized(){
  if(windowWidth>=770){
  document.getElementById('defaultCanvas0').style.width =windowWidth-17+"px";
    document.getElementById('defaultCanvas0').style.opacity ="1";
  document.getElementById('defaultCanvas0').style.height= windowHeight+"px";

  }
  else{
  document.getElementById('defaultCanvas0').style.height= 0+"px";
    document.getElementById('defaultCanvas0').style.opacity ="0";
  }
}


function setup() {
  canvas = createCanvas(windowWidth,windowHeight);
  document.getElementById('defaultCanvas0').style.margin ="0px 0px -10px 0";
  document.getElementById('defaultCanvas0').style.width =windowWidth-17+"px";


  if(windowWidth>=770){
    document.getElementById('defaultCanvas0').style.opacity ="1";
  }
  else{
    document.getElementById('defaultCanvas0').style.opacity ="0";
  }


  for( var i = 0; i< length; i++){
    stars.push(new Star());  
  }
}

function draw() {
      if (windowWidth>=770){
      background(0);
      speed =map(abs(mouseX-width/2),0,width/2,2,10);
      translate(width/2,height/2);
      for( var i = 0; i< length; i++){
        stars[i].show();
        stars[i].update();
      }
    }
}
  
class Star{
  constructor(){
    this.x = random(-width,width);
    this.y = random(-height,height);
    this.z = random(width);
    this.pz = this.z;
  }
  show(){
   fill(255);
   noStroke();
   this.sx = map(this.x/this.z,0,1,0,width);
   this.sy = map(this.y/this.z,0,1,0,height);
   this.r = map(this.z,0,width,8,0);
   ellipse(this.sx,this.sy,this.r,this.r);
   this.px = map(this.x/(this.pz+speed),0,1,0,width);
   this.py = map(this.y/(this.pz+speed),0,1,0,height);
   stroke(255);
   line(this.sx,this.sy,this.px,this.py);
   this.pz = this.z;
  }
  update(){
    this.z=this.z-speed;
    if(this.z<1){
      this.z=random(width);
      this.x = random(-width,width);
      this.y = random(-height,height);
      this.pz=this.z;
    }
  }
}
  
  
  
  
  
  
