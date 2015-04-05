

class MLP

  def initialize(ni, nh, no, weight_min=-2.0, weight_max=2.0)
    @w1, @w2, @dw1, @dw2 = [], [], [], []
    @ni, @nh, @no = ni, nh, no

    @vec_h, @vec_o, @vec_i = Array.new(nh, 0.0), Array.new(no, 0.0), Array.new(ni, 0.0)

    (0..ni).each do
      @w1 << (0...nh).collect{ rand(weight_min..weight_max) }
      @dw1 << Array.new(nh, 0.0)
    end

    (0...nh).each do
      @w2 << (0...no).collect{ rand(weight_min..weight_max) }
      @dw2 << Array.new(no, 0.0)
    end

  end

  def sigmoid(x)
    Math.sinh(x) / Math.cosh(x)  #Tanh(x) for some reason gives a runtime error.
  end

  def forward(vec_i)
    @vec_i = vec_i

    (0...@nh).each do |j|
      @vec_h[j] = sigmoid((0...@ni).inject(0){|sum, i| sum+(@vec_i[i] * @w1[i][j]) })
    end

    (0...@no).each do |j|
      @vec_o[j] = sigmoid((0...@nh).inject(0){|sum, i| sum + (@vec_h[i] * @w2[i][j]) })
    end

    @vec_o
  end

  def back_propagate(vec_t)

    output_deltas = vec_t.each_with_index.collect do |val_t, k|
      (1 - @vec_o[k]**2) * (val_t - @vec_o[k])
    end

    @vec_h.each_with_index do |val_h, i|
      output_deltas.each_with_index do |val_od, j|
        @dw2[i][j] = val_h * val_od
      end
    end

    hidden_deltas = @w2.each_with_index.collect do |w_val, i|
      (1 - @vec_h[i]**2) * output_deltas.each.with_index.inject(0){ |sum, (val_k, j)| sum+(val_k * w_val[j])}
    end

    @vec_i.each_with_index do |val_i, i|
      hidden_deltas.each_with_index do |val_hd, j|
        @dw1[i][j] = val_hd * val_i
      end
    end

    vec_t.each.with_index.inject(0){|sum, (val_t, k)| sum+(0.5 * (val_t-@vec_o[k])**2)}
  end

  def update_weights(lr)
    (0..@ni).each do |i|
      (0...@nh).each do |j|
        @w1[i][j] += @dw1[i][j]*lr
        @dw1[i][j] = 0.0
      end
    end

    (0...@nh).each do |i|
      (0...@no).each do |j|
        @w2[i][j] += @dw2[i][j]*lr
        @dw2[i][j] = 0.0
      end
    end
  end

end


def testXOR
  mlp = MLP.new(2, 2, 1)

  matrix = [[[0.0, 0.0], [0.0]],
            [[0.0, 1.0], [1.0]],
            [[1.0, 0.0], [1.0]],
            [[1.0, 1.0], [0.0]]]

  puts 'Training'
  puts '-------------------------'
  (0..10000).each do |i|
    error = 0.0

    matrix.each do |row|
      mlp.forward(row[0])
      error += mlp.back_propagate(row[1])
      mlp.update_weights(0.2)
    end
    puts 'Error: '+error.to_s if i % 100 == 0
  end

  puts
  puts 'Testing'
  puts '-------------------------'

  matrix.each do |row|
    p = mlp.forward(row[0])
    puts 'Predict:'+p[0].to_s
  end
end

def testRandSeq
  matrix = []
  (1..50).each do
    values = (0..3).collect{ rand(-1..1) }
    n = Math.sin(values.inject(0){|sum,x| sum + x })
    matrix << [values, [n]]
  end

  mlp = MLP.new(4, 7, 1, -0.5, 0.5)

  puts 'Training'
  puts '-------------------------'
  (0..400000).each do |i|
    error = 0.0

    matrix[0...40].each do |row|
      mlp.forward(row[0])
      error += mlp.back_propagate(row[1])
      mlp.update_weights(0.002)
    end
    puts i.to_s+' Error: '+error.to_s if i % 1000 == 0
  end

  puts
  puts 'Testing'
  puts '-------------------------'

  matrix[40..-1].each do |row|
    p = mlp.forward(row[0])
    puts 'Predict:'+p[0].round(4).to_s+' Actual: '+row[1][0].round(4).to_s+' '+row[0].to_s
  end
end

testRandSeq
