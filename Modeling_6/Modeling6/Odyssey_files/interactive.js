/*Load images*/
var troytext = new Image();
troytext.src = "./media/map/troytext.gif";
var troypict = new Image();
troypict.src = "./media/map/troypict.jpg";

var ciconestext = new Image();
ciconestext.src = "./media/map/ciconestext.gif";
var ciconespict = new Image();
ciconespict.src = "./media/map/ciconespict.jpg";

var lotostext = new Image();
lotostext.src = "./media/map/lotostext.gif";
var lotospict = new Image();
lotospict.src = "./media/map/lotospict.jpg";

var cyclopestext = new Image();
cyclopestext.src = "./media/map/cyclopestext.gif";
var cyclopespict = new Image();
cyclopespict.src = "./media/map/cyclopespict.jpg";

var aiolostext = new Image();
aiolostext.src = "./media/map/aiolostext.gif";
var aiolospict = new Image();
aiolospict.src = "./../globalmedia/pt.gif";

var laistrygonestext = new Image();
laistrygonestext.src = "./media/map/laistrygonestext.gif";
var laistrygonespict = new Image();
laistrygonespict.src = "./media/map/laistrygonespict.jpg";

var circetext = new Image();
circetext.src = "./media/map/circetext.gif";
var circepict = new Image();
circepict.src = "./media/map/circepict.jpg";

var underworldtext = new Image();
underworldtext.src = "./media/map/underworldtext.gif";
var underworldpict = new Image();
underworldpict.src = "./../globalmedia/pt.gif";

var sirenstext = new Image();
sirenstext.src = "./media/map/sirenstext.gif";
var sirenspict = new Image();
sirenspict.src = "./media/map/sirenspict.jpg";

var scyllatext = new Image();
scyllatext.src = "./media/map/scyllatext.gif";
var scyllapict = new Image();
scyllapict.src = "./media/map/scyllapict.jpg";

var heliostext = new Image();
heliostext.src = "./media/map/heliostext.gif";
var heliospict = new Image();
heliospict.src = "./media/map/heliospict.jpg";

var calypsotext = new Image();
calypsotext.src = "./media/map/calypsotext.gif";
var calypsopict = new Image();
calypsopict.src = "./media/map/calypsopict.jpg";

var phaeacianstext = new Image();
phaeacianstext.src = "./media/map/phaeacianstext.gif";
var phaeacianspict = new Image();
phaeacianspict.src = "./media/map/phaeacianspict.jpg";

var ithacatext = new Image();
ithacatext.src = "./media/map/ithacatext.gif";
var ithacapict = new Image();
ithacapict.src = "./media/map/ithacapict.jpg";

var legendtext = new Image();
legendtext.src = "./media/map/legendtext.gif";


function showText(whichOne) {

	document.images["theImage"].src = eval( "" + whichOne + "text.src" );

}




function showPicture(whichOne) {

	document.images["theImage"].src = eval( "" + whichOne + "pict.src");	

}