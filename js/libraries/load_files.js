p5.prototype.registerPreloadMethod('loadQD');
p5.prototype.registerPreloadMethod('loadMNIST');

p5.prototype.loadQD = function(file, callback) {
  var self = this;
  var data = {};
  var oReq = new XMLHttpRequest();
  oReq.open("GET", file, true);
  oReq.responseType = "arraybuffer";
  oReq.onload = function(oEvent) {
    var arrayBuffer = oReq.response;
    if (arrayBuffer) {
      data.bytes = new Uint8Array(arrayBuffer);
      if (callback) {
        callback(data);
      }
      self._decrementPreload();
    }
  }
  oReq.send(null);
  return data;
}

p5.prototype.loadMNIST = function(file, offset, callback) {
  var self = this;
  var data = {};
  var oReq = new XMLHttpRequest();
  oReq.open("GET", file, true);
  oReq.responseType = "arraybuffer";
  oReq.onload = function(oEvent) {
    var arrayBuffer = oReq.response;
    if (arrayBuffer) {
      data.bytes = new Uint8Array(arrayBuffer).slice(offset);
      if (callback) {
        callback(data);
      }
      self._decrementPreload();
    }
  }
  oReq.send(null);
  return data;
}
